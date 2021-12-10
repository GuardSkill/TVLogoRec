import os
import cv2
import glob
from tqdm import trange
#  把文件夹的png图像转换为jpg
dir='/Disk1/Dataset/TV_logo_data/train_01_V2/'
all_imgs=glob.glob(dir+'*.png')
bars = trange(len(all_imgs), leave=True)
for index in bars:
    buffer=cv2.imread(all_imgs[index])
    if buffer is not None:
        filename=all_imgs[index].split('.')[0]+'.jpg'
        # print(filename)
        cv2.imwrite(filename,buffer)
        os.remove(all_imgs[index])
