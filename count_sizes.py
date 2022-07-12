'''

Counting the distributions of all annotations.
Created by Q. Wang, 7.11.2022

'''

import json
import os
import openslide
import numpy as np 
from PIL import ImageDraw, Image
from scipy.ndimage.morphology import binary_fill_holes
import cv2
import matplotlib.pyplot as plt 

colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0)]

files = os.listdir('/home1/qiuliwang/Data/Glioma/20220502_labeled/')
json_files = []
for one_file in files:
    if '.json' in one_file:
        json_files.append(one_file)

class Json_Base:
    def __init__(self, path, case):
        print('Processing JSON file: ', path)
        self.path = path
        self.case = case
        self.load_dict = json.load(open(self.path, encoding = 'utf8'))
        self.first_level_keys = self.load_dict.keys()
        
        if '_via_settings' in self.first_level_keys and '_via_img_metadata' in self.first_level_keys and '_via_attributes' in self.first_level_keys:
            print('\tPreliminary check.')
        
    def get_wsi_info(self):
        image_info = self.load_dict['_via_img_metadata']
        keys = image_info.keys()
        assert len(keys) == 1
        
        basic_info = image_info[list(keys)[0]]
        self.filename = basic_info['filename']
        self.regions = basic_info['regions']
        return self.filename, self.regions
    
    def get_annotations(self):
        print('\tGet {0} annotations.'.format(len(self.regions)))
        anno_class = {}
        num = len(self.regions)
        for one_region in self.regions:
            one_class = one_region['region_attributes']['bone_marrow']
            if one_class not in anno_class.keys():
                anno_class[one_class] = []
            
            x_ = one_region['shape_attributes']['all_points_x']
            y_ = one_region['shape_attributes']['all_points_y']
            all_in = [x_, y_]
            anno_class[one_class].append(all_in)
        self.anno_class = anno_class
        return anno_class

        '''
        ['shape_attributes']['name']
        ['polygon', 'polyline']
        
        ['region_attributes']['bone_marrow']
        ['vessel', 'necrosis']
        '''

    def Paint(self, image, downsamples, mask_shape):
        anno_class = self.anno_class
        ori_image = image.copy()
        # img = image.load()
        draw = ImageDraw.Draw(image)
        mask = Image.new('L', mask_shape, 0)
        draw_mask = ImageDraw.Draw(mask)
        # # print(anno_class)
        color_id = 0

        x_ranges = []
        y_ranges = []
        for one_key in anno_class.keys():
            one_type = one_key
            if 'vessel' in one_type:
                color_id = 0
            elif 'necrosis' in one_type:
                color_id = 1
            else:
                color_id = 2
            print((one_type))
            # print(color_id)

            one_type_anno = anno_class[one_type]
            if 'vessel' in one_type:
                # Drawing each annotation
                for one_anno in one_type_anno:
                    x = np.array(one_anno[0])
                    y = np.array(one_anno[1])
                    x = x / downsamples#.astype(int)
                    y = y / downsamples#.astype(int)

                    x_max = x.max() + 100
                    x_min = x.min() - 100
                    y_max = y.max() + 100
                    y_min = y.min() - 100
                    x_ranges.append(x.max() - x.min())
                    y_ranges.append(y.max() - y.min())

                    sign = False
                    last_x = 0.0
                    last_y = 0.0
                    for one_x, one_y in zip(x, y):
                        # try:
                        if sign == False:
                            last_x = one_x
                            last_y = one_y
                            sign = True
                        else:
                            draw_mask.line([(last_x, last_y), (one_x, one_y)], fill = (255), width = 1)
                            last_x = one_x
                            last_y = one_y

                    crop = image.crop((x_min, y_min, x_max, y_max))  
                    crop.save(self.case + str((x_min, y_min, x_max, y_max)) + '_ori.png')
    
    
    def Counting_Annotations(self, annotations):
        anno_class = annotations
        counting_list = {}
        for one_key in anno_class.keys():
            one_type = one_key
            if 'vessel' in one_type:
                color_id = 0
            elif 'necrosis' in one_type:
                color_id = 1
            else:
                color_id = 2
            print((one_type))
            # print(color_id)
            downsamples = 1
            one_type_anno = anno_class[one_type]
            x_ranges = []
            y_ranges = []
            size_max_ = 0
            if 'vessel' in one_type:
                # Drawing each annotation
                for one_anno in one_type_anno:
                    x = np.array(one_anno[0])
                    y = np.array(one_anno[1])
                    x = x / downsamples#.astype(int)
                    y = y / downsamples#.astype(int)

                    x_max = x.max()
                    x_min = x.min()
                    y_max = y.max()
                    y_min = y.min()

                    x_range = x.max() - x.min()
                    y_range = y.max() - y.min()

                    size_max = x_range if x_range > y_range else y_range
                    if size_max > size_max_:
                        size_max_ = size_max

                    if size_max < 100:
                        sign = 100
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0
                    elif size_max < 200:
                        sign = 200
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0
                    elif size_max < 300:
                        sign = 300
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0
                    elif size_max < 400:
                        sign = 400
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0
                    elif size_max < 500:
                        sign = 500
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0    
                    elif size_max < 600:
                        sign = 600
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 700:
                        sign = 700
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 800:
                        sign = 800
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 900:
                        sign = 900
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 1000:
                        sign = 1000
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 1100:
                        sign = 1100
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0                      
                    elif size_max < 1200:
                        sign = 1200
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 1300:
                        sign = 1300
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0                      
                    elif size_max < 1400:
                        sign = 1400
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0   
                    elif size_max < 1500:
                        sign = 1500
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0                      
                    elif size_max < 1600:
                        sign = 1600
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 1700:
                        sign = 1700
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0                      
                    elif size_max < 1800:
                        sign = 1800
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 1900:
                        sign = 1900
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0  
                    elif size_max < 2000:
                        sign = 2000
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0                      
                    elif size_max < 3000:
                        sign = 3000
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0        
                    elif size_max < 4000:
                        sign = 4000
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0        
                    elif size_max < 5000:
                        sign = 5000
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0          
                    else:
                        sign = 5100
                        if sign in counting_list.keys():
                            counting_list[sign] += 1
                        else:
                            counting_list[sign] = 0 
                print('size_max_', size_max_)
        return counting_list     


json_path = '/home1/qiuliwang/Data/Glioma/svsLabel/'
wsi_path = '/home1/qiuliwang/Data/Glioma/svsData/'
json_files = os.listdir(json_path)

print('number of cases: ', len(json_files))

regions_of_all = {}
number_anno = 0
for one_json in json_files:
    case = one_json[ : len(one_json) - 5]
    ori = case
    print(one_json)
    print(case)
    one_json = json_path + one_json
    case = wsi_path + case
    slide = wsi_path + ori + '.svs'
    print(slide)

    slide = openslide.OpenSlide(slide)
    # print('Image dimension: ', slide.level_dimensions)
    print('Image downsamples: ', slide.level_downsamples)
    
    J = Json_Base(one_json, case)
    filename, regions_ = J.get_wsi_info()
    number_anno += len(regions_)
    regions = J.get_annotations()
    counting_list = J.Counting_Annotations(regions)

    for one_key in counting_list.keys():
        if one_key in regions_of_all.keys():
            regions_of_all[one_key] += counting_list[one_key]
        else:
            regions_of_all[one_key] = counting_list[one_key]


for one_key in regions_of_all.keys():
    print(one_key, regions_of_all[one_key])

print('number of all annotations: ', number_anno)
