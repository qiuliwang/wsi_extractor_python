from __future__ import division
import math
import numpy as np
from PIL import Image
import PIL
import cv2
# import and use one of 3 libraries PIL, cv2, or scipy in that order
USE_PIL = False
USE_CV2 = True
USE_SCIPY = False
from skimage.measure import label, regionprops

import os
import tqdm

USE_PIL = True

class ImageReadWrite(object):
    """expose methods for reading / writing images regardless of which
    library user has installed
    """

    def read(self, filename):
        if USE_PIL:
            color_im = PIL.Image.open(filename)
            # print(color_im.size)
            grey = color_im.convert('L')
            grey.save('grey.png')
            return np.array(grey, dtype=np.uint8)
        elif USE_CV2:
            img_grey = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(filename)
            # print(img.shape)
            channel_one = img[:,:,0]
            channel_two = img[:,:,1]
            channel_three = img[:,:,2]
            cv2.imwrite('grey3.png', channel_three)
            cv2.imwrite('grey2.png', channel_two)
            cv2.imwrite('grey1.png', channel_one)
            cv2.imwrite('grey.png', img_grey)

            return channel_one, channel_two, channel_three, img_grey
        elif USE_SCIPY:
            greyscale = True
            float_im = scipy.misc.imread(filename, greyscale)
            # convert float to integer for speed
            im = np.array(float_im, dtype=np.uint8)
            return im

    def write(self, filename, array):
        if USE_PIL:
            im = PIL.Image.fromarray(array)
            im.save(filename)
        elif USE_SCIPY:
            scipy.misc.imsave(filename, array)
        elif USE_CV2:
            cv2.imwrite(filename, array)



def remove_small_points(binary_img, threshold_area):
    """
    消除二值图像中面积小于某个阈值的连通域(消除孤立点)
    args:
        binary_img: 二值图
        threshold_area: 面积条件大小的阈值,大于阈值保留,小于阈值过滤
    return:
        resMatrix: 消除孤立点后的二值图
    """
    #输出二值图像中所有的连通域
    img_label, num = label(binary_img, connectivity=1, background=0, return_num=True) #connectivity=1--4  connectivity=2--8
    # print('+++', num, img_label)
    #输出连通域的属性，包括面积等
    props = regionprops(img_label) 
    # print('Number of props: ', len(props))
    ## adaptive threshold
    # props_area_list = sorted([props[i].area for i in range(len(props))]) 
    # threshold_area = props_area_list[-2]
    resMatrix = np.zeros(img_label.shape).astype(np.uint8)
    for i in range(0, len(props)):
        # print('--',props[i].area)
        # if props[i].area > 400 and props[i].area < 1000 and props[i].eccentricity > 0.7:
        if props[i].eccentricity < 0.75:
            tmp = (img_label == i + 1).astype(np.uint8)
            #组合所有符合条件的连通域
            resMatrix += tmp 
    resMatrix *= 255
    
    return resMatrix

def remove_fragment(img_pth, threshold = 60):
    img_gray = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE)
    # fixed thresh
    _, thresh = cv2.threshold(img_gray, 127, 255, type=cv2.THRESH_BINARY)
    ## method1
    thresh1 = thresh > 0
    # res_img1 = morphology.remove_small_objects(thresh1, 20)
    # res_img0 = morphology.remove_small_holes(thresh1, 20)
    ## method2
    # res_img1 = remove_small_points(thresh, 20)
    # res_img2 = remove_small_points(thresh, 40)
    res_img3 = remove_small_points(thresh, threshold)
    
   

    return res_img3

if __name__ == '__main__':
    k = 7
    base_dir = '512Crop/'
    ids = os.listdir(base_dir)
    imager = ImageReadWrite()
    for one_id in tqdm.tqdm(ids):
        files = os.listdir(os.path.join(base_dir, one_id))
        ori = []
        for one_file in files:
            if '_removefrag' in one_file:
                ori.append(one_file)
        print('Number of ori: ', len(ori))

        temp = 0

        for one_jpeg in tqdm.tqdm(ori):
            temp += 1
            filename_png = os.path.join(base_dir, one_id, one_jpeg)
            filename = filename_png.split('_removefrag.jpeg')[0]
            res = remove_fragment(filename_png, 60)
            savename = os.path.join(filename + '_test.jpeg')
            cv2.imwrite(savename, res)            