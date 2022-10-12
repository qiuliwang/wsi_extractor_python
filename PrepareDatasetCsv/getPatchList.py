import os
import tqdm
import numpy as np 
import csv

'''
Getting all patches and label weather or not there are vessels in this patch.
'''

data_path = '/home1/qiuliwang/Code/wsi_extractor_python/Glioma_Extracted_Patch_512/'
ids = os.listdir(data_path)

print('\nCounting files: \n')

all_files = []
for id in ids:
    files = os.listdir(os.path.join(data_path, id))
    for onefile in files:
        # print(os.path.join(data_path, id, onefile))
        all_files.append(os.path.join(data_path, id, onefile))

print('Number of files: ', len(all_files))

percent = 0.8
all_patches = []

print('\nCounting patches: \n')


labeled_patches_1 = []
labeled_patches_0 = []

for one_file in tqdm.tqdm(all_files):
    patch, filetype = one_file.split('.')
    if patch not in all_patches and 'mask' not in patch:
        all_patches.append(patch)
        mask = np.load(os.path.join(data_path, patch + '_mask.npy'))
        if mask.max() == 255:
            labeled_patches_1.append(os.path.join(data_path, patch) + ',1')
        else:
            labeled_patches_0.append(os.path.join(data_path, patch) + ',0')

print('Number of patches: ', len(all_patches))

print('\nPreparing masks: \n')

def writeCSV(filename, lines):
    with open(filename, "wb") as f:
        for line in lines:
            f.write((line + '\n').encode())

print('Number of 1 cases: ', len(labeled_patches_1))
print('Number of 0 cases: ', len(labeled_patches_0))

import random
random.shuffle(labeled_patches_1)
random.shuffle(labeled_patches_0)

labeled_patches_0 = labeled_patches_0[:len(labeled_patches_1)]
print('Number of 1 cases: ', len(labeled_patches_1))
print('Number of 0 cases: ', len(labeled_patches_0))

'''
Prepare data for labeled patches
'''
training_split_labeled = labeled_patches_1[ : int(len(labeled_patches_1) * percent)]
random.shuffle(training_split_labeled)

testing_split_labeled = labeled_patches_1[int(len(labeled_patches_1) * percent) : ]
random.shuffle(testing_split_labeled)

writeCSV('training_split_labeled.csv', training_split_labeled)
writeCSV('testing_split_labeled.csv', testing_split_labeled)


labeled_patches = labeled_patches_0 + labeled_patches_1
random.shuffle(labeled_patches)
writeCSV('all_data.csv', labeled_patches)

training_split = labeled_patches_1[ : int(len(labeled_patches_1) * percent)] + labeled_patches_0[ : int(len(labeled_patches_0) * percent)]
random.shuffle(training_split)

testing_split = labeled_patches_1[int(len(labeled_patches_1) * percent) : ] + labeled_patches_0[int(len(labeled_patches_0) * percent) : ]
random.shuffle(testing_split)

writeCSV('training_split.csv', training_split)
writeCSV('testing_split.csv', testing_split)

temp_training_split = labeled_patches_1[ : int(len(labeled_patches_1) * percent * 0.125)] + labeled_patches_0[ : int(len(labeled_patches_0) * percent * 0.125)]
random.shuffle(temp_training_split)

temp_testing_split = labeled_patches_1[int(len(labeled_patches_1) * percent * 0.125) : int(len(labeled_patches_1) * 0.125)] + labeled_patches_0[int(len(labeled_patches_0) * percent * 0.125) : int(len(labeled_patches_0) * 0.125)]
random.shuffle(temp_testing_split)

writeCSV('temp_training_split.csv', temp_training_split)
writeCSV('temp_testing_split.csv', temp_testing_split)