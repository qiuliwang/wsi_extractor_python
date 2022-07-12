import openslide
import os

'''
data_path = /home1/qiuliwang/Data/Glioma
/home1/qiuliwang/Data/Glioma/20220502_labeled
'''

data_path = '/home1/qiuliwang/Data/Glioma/data_svs/B201812997/'
files = os.listdir(data_path)
for one_file in files:
    slide = openslide.OpenSlide(data_path + one_file)

    dimensions = slide.level_dimensions
    print('Dimensions: ', dimensions)

    downsamples = slide.level_downsamples
    print('Downsamples: ', downsamples)

    image = slide.read_region((0,0),3,slide.level_dimensions[3])

    image.save(one_file + 'image.png')

# wsi_file = '/home1/qiuliwang/Data/Glioma/20220502_labeled/B202005122-2.svs'
# print(wsi_file)

# slide = openslide.OpenSlide(wsi_file)

# dimensions = slide.level_dimensions
# print('Dimensions: ', dimensions)

# downsamples = slide.level_downsamples
# print('Downsamples: ', downsamples)

# image = slide.read_region((0,0),2,slide.level_dimensions[2])

# print(type(image))

# image.save('image.png')