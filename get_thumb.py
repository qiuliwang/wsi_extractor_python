import json
import os
import openslide
import numpy as np 
from PIL import ImageDraw, Image
from scipy.ndimage.morphology import binary_fill_holes
import cv2
import matplotlib.pyplot as plt 
import tqdm

case_path = './'
cases = os.listdir(case_path)
print(cases)

# for one_case in cases:

wsi_path = os.path.join(case_path, 'B202105664-1.kfb')
# case_name = one_case.split('.svs')[0]

slide = openslide.OpenSlide(wsi_path)
level_index = 3

# print('Image dimension: ', slide.level_dimensions[level_index])
# print('Image downsamples: ', slide.level_downsamples[level_index])
# slide_pixels = slide.read_region((0, 0), level_index, slide.level_dimensions[level_index])
# slide_pixels = slide_pixels.convert('RGB')
# slide_pixels.save(case_name + '.png')
