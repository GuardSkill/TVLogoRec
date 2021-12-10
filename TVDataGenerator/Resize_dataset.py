import os
from math import floor

import cv2
import glob
from tqdm import trange
import numpy as np
def clamp_pixel(pv):
    if pv >= 255:
        return 255
    if pv < 0:
        return 0
    return pv

def gaussian_noise(image):
    height, width, channel = image.shape
    for row in range(height):
        for col in range(width):
            for c in range(channel):
                s = np.random.normal(0, 20, 3)
                b = image[row, col, 0]  # blue
                g = image[row, col, 1]  # green
                r = image[row, col, 2]  # red
                # print(row,col,b ,s[0],clamp(b + s[0]))
                image[row, col, 0] = clamp_pixel(b + s[0])
                image[row, col, 1] = clamp_pixel(g + s[1])
                image[row, col, 2] = clamp_pixel(r + s[2])
if __name__ == '__main__':
    # 把文件夹的png图像转换为jpg，并且一定概率降质量
    dir='/data/TVLogoDataset/TV_logo_data/train_4w_v3/'
    all_imgs=glob.glob(dir+'*.jpg')
    bars = trange(len(all_imgs), leave=True)
    KSIZE=5
    ### 4:3 800*600；1024*768；1152*864；1280*960；1280*1024都是4:3 的。
    for index in bars:
        flag=np.random.randint(0, 4)
        buffer=cv2.imread(all_imgs[index])
        h,w,c=buffer.shape
        if flag==0:  # resize到标清
            buffer=cv2.resize(buffer,(floor(w*0.375),floor(h*0.4444)))
            buffer = cv2.GaussianBlur(buffer, (KSIZE, KSIZE), 0)
            if np.random.randint(0, 2):
                gaussian_noise(buffer)
            buffer=cv2.resize(buffer,(w,h))
        elif flag==1:
            buffer=cv2.resize(buffer,(floor(w*0.666),floor(h*0.6666)))
            if np.random.randint(0, 2):
                gaussian_noise(buffer)
            buffer = cv2.resize(buffer, (w, h))
        if buffer is not None:
            # print('success')
            # print(filename)
            cv2.imwrite(all_imgs[index],buffer)

