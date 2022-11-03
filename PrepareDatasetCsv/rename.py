import os

data_path = '/home1/qiuliwang/Code/wsi_extractor_python/vessel_center/'
data_path2 = '/home1/qiuliwang/Code/wsi_extractor_python/vessel_center/'

files = os.listdir(data_path)

print('\nCounting files: \n')
vessels = []
for one_file in files:
    if 'mask' not in one_file:
        vessels.append(one_file)

print('Number of vessel images: ', len(vessels))

sign = 0
for vessel in vessels:
    id = vessel.split('_ori.jpeg')[0]
    mask1 = id + '_mask.jpeg'
    mask2 = id + '_mask2.jpeg'
    ori_path = os.path.join(data_path, vessel)
    target_path = os.path.join(data_path2, str(sign) + '.jpeg')

    ori_mask1_path = os.path.join(data_path, mask1)
    target_mask1_path = os.path.join(data_path2, str(sign) + '_mask1.jpeg')  

    ori_mask2_path = os.path.join(data_path, mask2)
    target_mask2_path = os.path.join(data_path2, str(sign) + '_mask2.jpeg')    

    os.rename(ori_path, target_path)
    os.rename(ori_mask1_path, target_mask1_path)
    os.rename(ori_mask2_path, target_mask2_path)
    sign += 1

# srcFile = './actwork/linkFile/allExtLinks - 副本.txt'
# dstFile = './actwork/linkFile/allExtLinks - copy.txt'
# try:
#     os.rename(srcFile,dstFile)
# except Exception as e:
#     print(e)
#     print('rename file fail\r\n')
# else:
#     print('rename file success\r\n')