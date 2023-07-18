'''
Get csv for vessel classification
'''

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


selected_dir1 = '/home1/qiuliwang/Code/wsi_extractor_python/vessel_center/selected/'
selected_dir2 = '/home1/qiuliwang/Code/wsi_extractor_python/vessel_center/selected2/'

temp_class1 = os.listdir(selected_dir1)
temp_class2 = os.listdir(selected_dir2)

class1 = []
class2 = []
for one_data in temp_class1:
    class1.append(one_data + ',' + str(1))

for one_data in temp_class2:
    class2.append(one_data + ',' + str(0))

random.shuffle(class1)
random.shuffle(class2)

percentage = 0.7

temp_training_list = class1[ : int(len(class1) * percentage)] + class2[ : int(len(class2) * percentage)]
temp_testing_list = class1[int(len(class1) * percentage) : ] + class2[int(len(class2) * percentage) : ]
random.shuffle(temp_training_list)
random.shuffle(temp_testing_list)
writeCSV('clas_training.csv', temp_training_list)
writeCSV('clas_testing.csv', temp_testing_list)