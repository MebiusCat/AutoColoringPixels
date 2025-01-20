import shutil
import os
import json

PWD = '.\\data\\mnist\\'
PWD_TMP = '.\\data\\mnist_tmp\\'
stats = {}
for i in range(-1, 100):
    num_path = f'{PWD}{i:02d}'
    if os.path.exists(num_path):
        stats[i] = len(os.listdir(num_path))


for i in range(-1, 100):
    num_path = f'{PWD_TMP}{i:02d}'
    before = stats.get(i, 0)

    if os.path.exists(num_path) and before < 1000:
        for elem in os.listdir(num_path):
            try:
                shutil.move(f'{num_path}\\{elem}', f'{PWD}{i:02d}\\{elem}')
            except:
                print(f'file {elem} copy error to {PWD}{i:02d}\\{elem}')
        if len(os.listdir(num_path)) == 0:
            shutil.rmtree(num_path)

stats = {}
for i in range(-1, 100):
    num_path = f'{PWD}{i:02d}'
    if os.path.exists(num_path):
        stats[i] = len(os.listdir(num_path))

with open('mnist_stat.json', 'w') as f:
    json.dump(stats, f, indent=4)