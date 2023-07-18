'''
Getting vessels from slices.
Each patch has a vessel in its center.
'''

import json
import os
import openslide
import numpy as np 
from PIL import ImageDraw, Image
from scipy.ndimage.morphology import binary_fill_holes
import cv2
import matplotlib.pyplot as plt 
import tqdm
import threading

colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0)]

# files = os.listdir('/home1/qiuliwang/Data/Glioma/20220502_labeled/')
# json_files = []
# for one_file in files:
#     if '.json' in one_file:
#         json_files.append(one_file)

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
                for one_anno in tqdm.tqdm(one_type_anno):
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
                    xy_list = []
                    for one_x, one_y in zip(x, y):
                        xy_list.append((one_x, one_y))

                    draw.polygon(xy_list, fill=None, outline=(255))
                    draw_mask.polygon(xy_list, fill=(255), outline=None)


                        # except:
                        #     print(color_id)

                    #crop a single annotation
                    # crop = image.crop((x_min, y_min, x_max, y_max))  
                    # crop.save(self.case + str((x_min, y_min, x_max, y_max)) + '_ori.png')
                    # crop_mask = mask.crop((x_min, y_min, x_max, y_max)) 
                    # ImageDraw.floodfill(crop_mask, (0, 0), (255))
                    # mask_npy = np.array(crop_mask) 
                    # np.save(self.case + '_mask.npy', mask_npy)
                    # crop_mask.save(self.case + str((x_min, y_min, x_max, y_max)) + '_mask.png')
                    # np.save(self.case + str((x_min, y_min, x_max, y_max)) + '_mask.npy', mask_npy)

            # save all versions

            image.save(self.case + '_overlay.png') 
            print('Saving mask...')
            # ImageDraw.floodfill(mask, (0, 0), (255))
            mask_npy = np.array(mask) 

            np.save(self.case + '_mask.npy', mask_npy)
            mask.convert('RGBA')
            mask.save(self.case + '_mask.png')
            ori_image.save(self.case + '.png')

            # # plt.hist(x_ranges)
            # # plt.savefig("x_ranges.jpg")          
            # plt.hist(y_ranges)
            # plt.savefig("y_ranges.jpg")

    def Paint_noImage(self, downsamples, mask_shape):
        anno_class = self.anno_class
        # ori_image = image.copy()
        # img = image.load()
        # draw = ImageDraw.Draw(image)
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
                for one_anno in tqdm.tqdm(one_type_anno):
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
                    xy_list = []
                    for one_x, one_y in zip(x, y):
                        xy_list.append((one_x, one_y))
                    
                    draw_mask.polygon(xy_list, fill=(255), outline=None)

                        # except:
                        #     print(color_id)

                    #crop a single annotation
                    # crop = image.crop((x_min, y_min, x_max, y_max))  
                    # crop.save(self.case + str((x_min, y_min, x_max, y_max)) + '_ori.png')
                    # crop_mask = mask.crop((x_min, y_min, x_max, y_max)) 
                    # ImageDraw.floodfill(crop_mask, (0, 0), (255))
                    # mask_npy = np.array(crop_mask) 
                    # np.save(self.case + '_mask.npy', mask_npy)
                    # crop_mask.save(self.case + str((x_min, y_min, x_max, y_max)) + '_mask.png')
                    # np.save(self.case + str((x_min, y_min, x_max, y_max)) + '_mask.npy', mask_npy)

            # save all versions

            # image.save(self.case + '_overlay.png') 
            print('Saving mask...')
            # ImageDraw.floodfill(mask, (0, 0), (255))
            mask_npy = np.array(mask) 

            np.save(self.case + '_mask.npy', mask_npy)
            # mask.convert('RGBA')
            # mask.save(self.case + '_mask.png')
            # ori_image.save(self.case + '.png')

            # # plt.hist(x_ranges)
            # # plt.savefig("x_ranges.jpg")          
            # plt.hist(y_ranges)
            # plt.savefig("y_ranges.jpg")

    def process_vessel_single(self, ori_image, one_type_anno, mask_shape, downsamples):
        for one_anno in tqdm.tqdm(one_type_anno):
            x = np.array(one_anno[0])
            y = np.array(one_anno[1])
            x = x / downsamples#.astype(int)
            y = y / downsamples#.astype(int)

            x_max = int(x.max())
            x_min = int(x.min())
            y_max = int(y.max())
            y_min = int(y.min())

            sign = False
            last_x = 0.0
            last_y = 0.0
            xy_list = []
            for one_x, one_y in zip(x, y):
                xy_list.append((one_x, one_y))
            
            mask = Image.new('L', mask_shape, 0)
            draw_mask = ImageDraw.Draw(mask)
            draw_mask.polygon(xy_list, fill=None, outline=(255))
            # draw_mask.polygon(xy_list, fill=(255), outline=None)

            # mask2 = Image.new('L', mask_shape, 0)
            # draw_mask2 = ImageDraw.Draw(mask2)
            # draw.polygon(xy_list, fill=None, outline=(255))
            # draw_mask2.polygon((x_max, y_max, x_max, y_min, x_min, y_min, x_min, y_max), fill=(255), outline=None)

                # except:
                #     print(color_id)

            # crop a single annotation
            sign = 1024
            # if abs(x_max - x_min) <= sign and abs(y_max - y_min) <= sign and abs(x_max - x_min) > sign / 2 and abs(y_max - y_min) > sign / 2 :
            if abs(x_max - x_min) <= sign and abs(y_max - y_min) <= sign :
                mid_x = x_min + (abs(x_max - x_min) / 2)
                mid_y = y_min + (abs(y_max - y_min) / 2)

                crop = ori_image.crop((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2))  
                crop.save('vessel_images/ori/' + self.case + str((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2)) + '_ori.jpeg')
               
                crop_mask = mask.crop((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2)) 
                # crop_mask2 = mask2.crop((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2)) 

                ImageDraw.floodfill(crop_mask, (0, 0), (255))
                # mask_npy = np.array(crop_mask) 
                # np.save('vessel_images/mask1/' + self.case + '_mask.npy', mask_npy)

                crop_mask.save('vessel_images/mask1/' + self.case + str((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2)) + '_mask.jpeg')
                # crop_mask2.save('vessel_images/mask2/' + self.case + str((mid_x - sign/2, mid_y - sign/2, mid_x + sign/2, mid_y + sign/2)) + '_mask2.jpeg')
                # np.save('vessel_images/mask2/' + self.case + str((x_min, y_min, x_max, y_max)) + '_mask.npy', mask_npy)

    def Paint_vessels(self, image, downsamples, mask_shape):
        anno_class = self.anno_class
        ori_image = image.copy()
        # img = image.load()
        draw = ImageDraw.Draw(image)

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
                # self.process_vessel_single(ori_image, one_type_anno, mask_shape, downsamples)
                
                ttt = len(one_type_anno) // 2
                thread1 = threading.Thread(name='t1',target= self.process_vessel_single,args=(ori_image, one_type_anno[ : ttt], mask_shape, downsamples))
                thread2 = threading.Thread(name='t2',target= self.process_vessel_single,args=(ori_image, one_type_anno[ttt : ], mask_shape, downsamples))
                # thread3 = threading.Thread(name='t3',target= self.process_vessel_single,args=(ori_image, one_type_anno[ttt + ttt : ], mask_shape, downsamples))
                thread1.start()   #启动线程1
                thread2.start()   #启动线程2
                # thread3.start()   #启动线程2


cases = os.listdir('/home1/qiuliwang/Data/Glioma/svsData/')
count = 0
sign = 0
print(len(cases))

training_list = cases
for one_case in cases:
    if one_case in training_list[15:]:
        print(one_case)
        wsi_path = '/home1/qiuliwang/Data/Glioma/svsData/' + one_case
        json_path = '/home1/qiuliwang/Data/Glioma/svsLabel/' + one_case[ : len(one_case) - 4] + '.json'

        J = Json_Base(json_path, one_case[ : len(one_case) - 4])
        filename, regions = J.get_wsi_info()
        count += len(regions)
        regions = J.get_annotations()
        
        print(one_case + ', ' + str(len(J.regions)))
        if len(J.regions) >= 1000:
            training_list.append(one_case + ', ' + str(len(J.regions)))

        slide = openslide.OpenSlide(wsi_path)
        level_index = 0

        print('Image dimension: ', slide.level_dimensions[level_index])
        print('Image downsamples: ', slide.level_downsamples[level_index])
        slide_pixels = slide.read_region((0, 0), level_index, slide.level_dimensions[level_index])
        slide_pixels = slide_pixels.convert('RGB')
        mask_shape = slide.level_dimensions[level_index]
        print('Shape: ', slide.level_dimensions[level_index])

        J.Paint_vessels(slide_pixels, slide.level_downsamples[level_index], mask_shape)

print("Number of all annotations: ", count)

# print(training_list)