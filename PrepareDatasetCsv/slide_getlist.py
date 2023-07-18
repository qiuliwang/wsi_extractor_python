import os
import tqdm
import numpy as np 
import csv
import random

'''
Getting all patches and label weather or not there are vessels in this patch.
'''

def writeCSV(filename, lines):
    with open(filename, "wb") as f:
        for line in lines:
            f.write((line + '\n').encode())

training_list = ['B201902483-5', 'B202005279-3', 'B201812997-1', 'B202005499-4', 'B201911880-10', 'B201812997-2', 'B201814368-5', 'B202005122-2', 'B202105664-14', 'B201813472-2', 'B202104270-3']

testing_list = ['B202012437-4', 'B202105664-3']

data_path = '/home1/qiuliwang/Code/wsi_extractor_python/vessel_center/'
files = os.listdir(data_path)

print('\nCounting files: \n')
vessels = []
for one_file in files:
    if 'mask' not in one_file:
        vessels.append(os.path.join(data_path, one_file))

print('Number of vessel images: ', len(vessels))

percentage = 0.7
random.shuffle(vessels)

training_data = vessels[ : int(len(vessels) * percentage)]
testing_data = vessels[int(len(vessels) * percentage) : ]
print('Number of training images: ', len(training_data))
print('Number of testing images: ', len(testing_data))

training_vessel = []
training_mask1 = []
training_mask2 = []
for vessel in training_data:
    id = vessel.split('.jpeg')[0]
    vessel_ = id +'.jpeg '
    mask1 = id + '_mask.jpeg '
    mask2 = id + '_mask2.jpeg '
    training_vessel.append(vessel_)
    training_mask1.append(mask1)
    training_mask2.append(mask2)

writeCSV('training_data.csv', training_vessel)
writeCSV('training_mask1.csv', training_mask1)
writeCSV('training_mask2.csv', training_mask2)

testing_vessel = []
testing_mask1 = []
testing_mask2 = []
for vessel in testing_data:
    id = vessel.split('.jpeg')[0]
    vessel_ = id +'.jpeg ' 
    mask1 = id + '_mask.jpeg '
    mask2 = id + '_mask2.jpeg '
    testing_vessel.append(vessel_)
    testing_mask1.append(mask1)
    testing_mask2.append(mask2)

writeCSV('testing_data.csv', testing_vessel)
writeCSV('testing_mask1.csv', testing_mask1)
writeCSV('testing_mask2.csv', testing_mask2)
