import os
import tqdm
import numpy as np 
import csv
from PIL import Image
'''
Getting all patches and label weather or not there are vessels in this patch.
'''
data_path = '/home1/qiuliwang/Code/wsi_extractor_python/Glioma_Extracted_Patch_512/'
ids = os.listdir(data_path)

for id in ids:  
    all_files = os.listdir(os.path.join(data_path, id))

    all_patches = []

    print('\nCounting patches: \n')
    for one_file in tqdm.tqdm(all_files):
        patch, filetype = one_file.split('.')
        if patch not in all_patches and 'mask' not in patch:
            all_patches.append(patch)

    print('Number of patches: ', len(all_patches))

    print('\nPreparing masks: \n')
    labeled_patches = []
    for one_patch in tqdm.tqdm(all_patches):
        mask = np.load(os.path.join(data_path, id, one_patch + '_mask.npy'))

        mask = np.where(mask == True, 255.0, 0.0)
        np.save(os.path.join(data_path, id, one_patch + '_mask.npy'), mask)
        mask = Image.fromarray(mask)
        mask = mask.convert('L')
        mask.save(os.path.join(data_path, id, one_patch + '_mask.jpeg'))