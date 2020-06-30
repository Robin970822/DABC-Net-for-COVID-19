import os
import config
import numpy as np

from tqdm import tqdm

for file in tqdm(config.npy_path):
    filename = os.path.basename(file)
    data = np.load(file)
    fliped = np.flip(data, axis=1)
    out_filename = os.path.join(config.rotate_root, filename)
    print(out_filename)
    np.save(out_filename, fliped)
