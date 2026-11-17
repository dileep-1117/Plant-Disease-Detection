import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

model = keras.models.load_model('plant_disease_model.keras', compile=False)
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

test_dir = Path('dataset_test')
all_images = list(test_dir.glob('*/*.*'))

results = []

for img_path in sorted(all_images):
    actual_class = img_path.parent.name
    actual_plant = actual_class.split('___')[0].replace('_', ' ').strip()
    
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    t = tf.convert_to_tensor(arr)
    t = tf.image.resize(t, (96, 96))
    t = tf.cast(t, tf.float32) / 255.0
    t = tf.expand_dims(t, axis=0)

    preds = model(t, training=False).numpy()[0]
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    pred_plant = pred_class.split('___')[0].replace('_', ' ').strip()
    conf = float(preds[pred_idx])

    top5_indices = np.argsort(preds)[::-1][:5]
    top5 = [(int(i), class_names[int(i)], float(preds[int(i)])) for i in top5_indices]

    results.append({
        'path': str(img_path),
        'filename': img_path.name,
        'actual_class': actual_class,
        'actual_plant': actual_plant,
        'pred_class': pred_class,
        'pred_plant': pred_plant,
        'pred_idx': pred_idx,
        'conf': conf,
        'is_plant_correct': (actual_plant == pred_plant),
        'is_disease_correct': (actual_class == pred_class),
        'top5': top5,
        'raw_probs': preds.tolist()
    })

print(f"Total evaluated images: {len(results)}")
plant_correct = sum(1 for r in results if r['is_plant_correct'])
disease_correct = sum(1 for r in results if r['is_disease_correct'])
print(f"Plant Species Accuracy: {plant_correct}/{len(results)} ({plant_correct/len(results)*100:.2f}%)")
print(f"Exact Disease Accuracy: {disease_correct}/{len(results)} ({disease_correct/len(results)*100:.2f}%)")

# Per-plant accuracy
plants = sorted(list(set(r['actual_plant'] for r in results)))
print("\n=== PER-PLANT ACCURACY ===")
for p in plants:
    p_results = [r for r in results if r['actual_plant'] == p]
    p_corr = sum(1 for r in p_results if r['is_plant_correct'])
    d_corr = sum(1 for r in p_results if r['is_disease_correct'])
    print(f"  {p:<12}: Plant Match = {p_corr}/{len(p_results)} ({p_corr/len(p_results)*100:.1f}%) | Disease Match = {d_corr}/{len(p_results)} ({d_corr/len(p_results)*100:.1f}%)")

print("\n=== MISCLASSIFICATIONS ===")
for r in results:
    if not r['is_disease_correct']:
        print(f"  File: {r['filename']}")
        print(f"    Actual: {r['actual_class']} (Plant: {r['actual_plant']})")
        print(f"    Pred  : {r['pred_class']} (Plant: {r['pred_plant']}) | Conf: {r['conf']*100:.2f}%")
        print(f"    Top 3 : {[(k[1], f'{k[2]*100:.1f}%') for k in r['top5'][:3]]}")
