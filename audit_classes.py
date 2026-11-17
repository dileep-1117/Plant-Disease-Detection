import urllib.request
import json

with open('class_names.json', 'r') as f:
    local_classes = json.load(f)

req = urllib.request.Request('https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color', headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    repo_folders = [item['name'] for item in json.loads(resp.read().decode())]

print(f"{'IDX':<4} | {'LOCAL class_names.json':<45} | {'REPO FOLDERS (Alphabetical)':<45} | Match")
print('-' * 110)
for i in range(38):
    loc = local_classes[i]
    rep = repo_folders[i]
    match = 'YES' if loc == rep else 'MISMATCH'
    print(f"{i:2d}   | {loc:<45} | {rep:<45} | {match}")
