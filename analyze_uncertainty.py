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

correct_confs = []
incorrect_confs = []
ambiguity_margins = [] # diff between rank 1 and rank 2

for img_path in sorted(all_images):
    actual_class = img_path.parent.name
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    t = tf.convert_to_tensor(arr)
    t = tf.image.resize(t, (96, 96))
    t = tf.cast(t, tf.float32) / 255.0
    t = tf.expand_dims(t, axis=0)

    preds = model(t, training=False).numpy()[0]
    top_indices = np.argsort(preds)[::-1]
    top1_conf = float(preds[top_indices[0]])
    top2_conf = float(preds[top_indices[1]])
    margin = top1_conf - top2_conf
    ambiguity_margins.append((margin, top1_conf, actual_class == class_names[top_indices[0]]))

    if actual_class == class_names[top_indices[0]]:
        correct_confs.append(top1_conf)
    else:
        incorrect_confs.append(top1_conf)

print(f"Correct predictions count: {len(correct_confs)}")
print(f"  Min conf: {min(correct_confs):.4f}, Mean conf: {np.mean(correct_confs):.4f}, Median conf: {np.median(correct_confs):.4f}")
print(f"Incorrect predictions count: {len(incorrect_confs)}")
print(f"  Min conf: {min(incorrect_confs):.4f}, Mean conf: {np.mean(incorrect_confs):.4f}, Median conf: {np.median(incorrect_confs):.4f}")

print("\nAmbiguity & Margin Analysis for Misclassifications:")
for margin, top1, is_corr in ambiguity_margins:
    if not is_corr:
        print(f"  Misclassified: Top1 Conf={top1*100:.2f}%, Top1-Top2 Margin={margin*100:.2f}%")
