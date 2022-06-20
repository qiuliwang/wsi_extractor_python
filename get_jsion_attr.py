import json
import os
import openslide
import numpy as np 
from PIL import ImageDraw, Image
from scipy.ndimage.morphology import binary_fill_holes
import cv2

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
        # print(mask.size)
        # # print(anno_class)
        color_id = 0

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

                            # draw.ellipse((one_x - 3, one_y - 3, one_x + 3, one_y + 3), fill=colors[color_id])
                            draw.line([(last_x, last_y), (one_x, one_y)], fill = colors[color_id], width = 3)
                            draw_mask.line([(last_x, last_y), (one_x, one_y)], fill = (255), width = 1)
                            last_x = one_x
                            last_y = one_y
                                # img[one_x, one_y] = colors[color_id] 
                                # img[one_x - 1, one_y] = colors[color_id]
                                # img[one_x + 1, one_y] = colors[color_id]
                                # img[one_x - 2, one_y] = colors[color_id]
                                # img[one_x + 2, one_y] = colors[color_id]
                                # img[one_x, one_y - 1] = colors[color_id]
                                # img[one_x, one_y + 1] = colors[color_id]
                                # img[one_x, one_y - 2] = colors[color_id]
                                # img[one_x, one_y + 2] = colors[color_id]
                        # except:
                        #     print(color_id)

                    #crop a single annotation
                    crop = image.crop((x_min, y_min, x_max, y_max))  
                    crop.save(self.case + str((x_min, y_min, x_max, y_max)) + '_mask.png')

            # save all versions
            # image.save(self.case + '_overlay.png') 
            # ImageDraw.floodfill(mask, (0, 0), (255))
            # mask.convert('RGBA')
            # mask.save(self.case + '_mask.png')
            # ori_image.save(self.case + '.png')


case = 'B202005122-2'
json_path = '/home1/qiuliwang/Data/Glioma/20220502_labeled/' + case + '.json'
wsi_path = '/home1/qiuliwang/Data/Glioma/20220502_labeled/' + case + '.svs'
J = Json_Base(json_path, case)
filename, regions = J.get_wsi_info()
regions = J.get_annotations()

slide = openslide.OpenSlide(wsi_path)
level_index = 0

print('Image dimension: ', slide.level_dimensions)
print('Image downsamples: ', slide.level_downsamples[level_index])
slide_pixels = slide.read_region((0, 0), level_index, slide.level_dimensions[level_index])
slide_pixels = slide_pixels.convert('RGB')
mask_shape = slide_pixels.size
print('Shape: ', mask_shape)

J.Paint(slide_pixels, slide.level_downsamples[level_index], mask_shape)