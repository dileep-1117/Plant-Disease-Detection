import urllib.request
import json
import os
from pathlib import Path

corn_folders = [
    ('Corn_(maize)___Common_rust_', 'Corn___Common_rust'),
    ('Corn_(maize)___Northern_Leaf_Blight', 'Corn___Northern_Leaf_Blight'),
    ('Corn_(maize)___healthy', 'Corn___healthy')
]

base_dir = Path('dataset_test')
base_dir.mkdir(exist_ok=True)
base_api = 'https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/'

for repo_folder, local_name in corn_folders:
    target_dir = base_dir / local_name
    target_dir.mkdir(exist_ok=True)
    try:
        url = base_api + urllib.parse.quote(repo_folder)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
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
            print(f'Fetched {count} images for {local_name}')
    except Exception as e:
        print(f'Error fetching {repo_folder}: {e}')
