import urllib.request
import json
import os
from pathlib import Path

classes_to_fetch = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight'
]

base_dir = Path('dataset_test')
base_dir.mkdir(exist_ok=True)
base_api = 'https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/'

for cls_name in classes_to_fetch:
    target_dir = base_dir / cls_name
    target_dir.mkdir(exist_ok=True)
    try:
        req = urllib.request.Request(base_api + cls_name, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            items = json.loads(resp.read().decode())
            count = 0
            for item in items:
                fname = item['name']
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    dst = target_dir / fname
                    if not dst.exists():
                        urllib.request.urlretrieve(item['download_url'], str(dst))
                    count += 1
                    if count >= 3:
                        break
            print(f'Fetched {count} images for {cls_name}')
    except Exception as e:
        print(f'Error fetching {cls_name}: {e}')
