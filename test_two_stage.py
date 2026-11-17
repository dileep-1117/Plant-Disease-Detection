import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

model = keras.models.load_model('plant_disease_model.keras', compile=False)
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Group classes by plant species
from collections import defaultdict
plant_to_classes = defaultdict(list)
class_to_plant = {}

for idx, c in enumerate(class_names):
    p = c.split('___')[0].replace('_', ' ').replace(',', '').strip()
    if p.lower() == 'pepper bell':
        p = 'Pepper (Bell)'
    elif p.lower() == 'cherry (including sour)':
        p = 'Cherry'
    elif p.lower() == 'corn (maize)':
        p = 'Corn'
    plant_to_classes[p].append(idx)
    class_to_plant[idx] = p

print("Plant species count:", len(plant_to_classes))
for p, idxs in sorted(plant_to_classes.items()):
    print(f"  {p:<15} ({len(idxs)} classes): {[class_names[i].split('___')[1] for i in idxs]}")

test_dir = Path('dataset_test')
all_images = list(test_dir.glob('*/*.*'))

two_stage_results = []

for img_path in sorted(all_images):
    actual_class = img_path.parent.name
    actual_plant_raw = actual_class.split('___')[0].replace('_', ' ').replace(',', '').strip()
    if actual_plant_raw.lower() == 'pepper bell':
        actual_plant = 'Pepper (Bell)'
    else:
        actual_plant = actual_plant_raw

    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    t = tf.convert_to_tensor(arr)
    t = tf.image.resize(t, (96, 96))
    t = tf.cast(t, tf.float32) / 255.0
    t = tf.expand_dims(t, axis=0)

    preds = model(t, training=False).numpy()[0]

    # Stage 1: Plant identification by marginalizing over diseases
    plant_probs = {}
    for p, idxs in plant_to_classes.items():
        plant_probs[p] = float(np.sum(preds[idxs]))

    sorted_plants = sorted(plant_probs.items(), key=lambda x: x[1], reverse=True)
    pred_plant, plant_conf = sorted_plants[0]

    # Stage 2: Disease identification conditioned on predicted plant
    candidate_idxs = plant_to_classes[pred_plant]
    candidate_probs = {idx: float(preds[idx]) for idx in candidate_idxs}
    sum_candidate = sum(candidate_probs.values())
    
    if sum_candidate > 0:
        cond_probs = {idx: p / sum_candidate for idx, p in candidate_probs.items()}
    else:
        cond_probs = candidate_probs

    sorted_cond = sorted(cond_probs.items(), key=lambda x: x[1], reverse=True)
    best_disease_idx, cond_disease_conf = sorted_cond[0]
    best_disease_class = class_names[best_disease_idx]
    
    # Flat top-1
    flat_idx = int(np.argmax(preds))
    flat_class = class_names[flat_idx]
    flat_plant = class_to_plant[flat_idx]

    two_stage_results.append({
        'filename': img_path.name,
        'actual_class': actual_class,
        'actual_plant': actual_plant,
        'flat_pred_class': flat_class,
        'flat_pred_plant': flat_plant,
        'pred_plant': pred_plant,
        'plant_conf': plant_conf,
        'pred_disease_class': best_disease_class,
        'cond_disease_conf': cond_disease_conf,
        'flat_conf': float(preds[flat_idx]),
        'plant_correct': (actual_plant.lower() == pred_plant.lower()),
        'disease_correct': (actual_class.lower() == best_disease_class.lower()),
    })

print("\n" + "=" * 80)
print(f"TWO-STAGE PREDICTION EVALUATION ({len(two_stage_results)} images)")
print("=" * 80)
p_corr = sum(1 for r in two_stage_results if r['plant_correct'])
d_corr = sum(1 for r in two_stage_results if r['disease_correct'])
print(f"Stage 1 (Plant Species) Accuracy : {p_corr}/{len(two_stage_results)} ({p_corr/len(two_stage_results)*100:.2f}%)")
print(f"Stage 2 (Disease Condition) Acc  : {d_corr}/{len(two_stage_results)} ({d_corr/len(two_stage_results)*100:.2f}%)")

# Compare with flat prediction
flat_p_corr = sum(1 for r in two_stage_results if r['actual_plant'].lower() == r['flat_pred_plant'].lower())
flat_d_corr = sum(1 for r in two_stage_results if r['actual_class'].lower() == r['flat_pred_class'].lower())
print(f"Flat Baseline Plant Accuracy     : {flat_p_corr}/{len(two_stage_results)} ({flat_p_corr/len(two_stage_results)*100:.2f}%)")
print(f"Flat Baseline Disease Accuracy   : {flat_d_corr}/{len(two_stage_results)} ({flat_d_corr/len(two_stage_results)*100:.2f}%)")
