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

print(f"Total test images: {len(all_images)}")
print(f"{'IMAGE LABEL':<30} | {'PRED IDX':<8} | {'LOCAL CLASS NAME':<35} | {'CONF':<8}")
print('-' * 95)

for img_path in sorted(all_images):
    actual_folder = img_path.parent.name
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    t = tf.convert_to_tensor(arr)
    t = tf.image.resize(t, (96, 96))
    t = tf.cast(t, tf.float32) / 255.0
    t = tf.expand_dims(t, axis=0)

    preds = model(t, training=False).numpy()[0]
    best_idx = int(np.argmax(preds))
    conf = float(preds[best_idx]) * 100

    print(f"{actual_folder:<30} | {best_idx:<8} | {class_names[best_idx]:<35} | {conf:.2f}%")
