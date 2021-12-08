import os
import cv2
import glob
from tqdm import trange
import numpy as np

def print_class(all_imgs):
    names=[]
    for img in all_imgs:
        names.append(os.path.basename(img).split('.')[0])
    print(names)
#  把文件夹的png图像转换为jpg
def crop_save_logo(new_dir, all_imgs, right_up_corn_logos):
    # bars = trange(len(all_imgs), leave=True)
    for index in range(len(all_imgs)):
        buffer = cv2.imread(all_imgs[index], cv2.IMREAD_UNCHANGED)
        # print(all_imgs[index])
        B, G, R, A = cv2.split(buffer)
        h, w, c = buffer.shape
        if buffer is not None:
            border = 8
            A = A[border:600, border:600]
            ret, binary = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # c = max(cnts, key=cv2.contourArea)
            extLeft = 600
            extRight = 0
            extTop = 600
            extBot = 0
            if len(cnts) > 0:
                delta = 3
                for c in cnts:
                    Left = tuple(c[c[:, :, 0].argmin()][0])[0]
                    extLeft = Left if Left < extLeft else extLeft
                    Right = tuple(c[c[:, :, 0].argmax()][0])[0]
                    extRight = Right if Right > extRight else extRight
                    Top = tuple(c[c[:, :, 1].argmin()][0])[1]
                    extTop = Top if Top < extTop else extTop
                    Bot = tuple(c[c[:, :, 1].argmax()][0])[1]
                    extBot = Bot if Bot > extBot else extBot
                crop_buffer = buffer[max(0, (extTop - delta)) + border:(extBot + delta) + border,
                              max(0, (extLeft - delta)) + border:(extRight + delta) + border]
                filename = new_dir + os.path.basename(all_imgs[index])
                cv2.imwrite(filename, crop_buffer)
                print(filename)
    all_imgs = right_up_corn_logos
    # bars = trange(len(all_imgs), leave=True)
    for index in range(len(all_imgs)):
        buffer = cv2.imread(all_imgs[index], cv2.IMREAD_UNCHANGED)
        # print(all_imgs[index])
        B, G, R, A = cv2.split(buffer)
        h, w, c = buffer.shape
        if buffer is not None:
            border = 8
            # A = A[border:h // 2, border:w // 2]
            A = A[border:h // 2, border + w // 2:]  # 星空卫视HD.png
            ret, binary = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # c = max(cnts, key=cv2.contourArea)
            extLeft = w // 2
            extRight = 0
            extTop = h // 2
            extBot = 0
            if len(cnts) > 0:
                delta = 3
                for c in cnts:
                    Left = tuple(c[c[:, :, 0].argmin()][0])[0]
                    extLeft = Left if Left < extLeft else extLeft
                    Right = tuple(c[c[:, :, 0].argmax()][0])[0]
                    extRight = Right if Right > extRight else extRight
                    Top = tuple(c[c[:, :, 1].argmin()][0])[1]
                    extTop = Top if Top < extTop else extTop
                    Bot = tuple(c[c[:, :, 1].argmax()][0])[1]
                    extBot = Bot if Bot > extBot else extBot
                # crop_buffer = buffer[max(0,(extTop-delta))+border:(extBot+delta)+border,
                #               max(0,(extLeft-delta))+border+w // 2:(extRight+delta)+border+w // 2]
                # for  星空卫视HD.png
                crop_buffer = buffer[max(0, (extTop - delta)) + border:(extBot + delta) + border,
                              max(0, (extLeft - delta)) + border + w // 2:(extRight + delta) + border + w // 2]
                filename = new_dir + os.path.basename(all_imgs[index])
                cv2.imwrite(filename, crop_buffer)
                print(filename)
            # print(filename)
            # cv2.imwrite(filename,buffer)

dir = '/Disk1/Dataset/TV_logo_data/train_01_V2/'
dir = '/Disk1/Dataset/TV_logo_data/全图版本/'
new_dir = '/Disk1/Dataset/TV_logo_data/TV_logo_Crop/'
os.makedirs(new_dir, exist_ok=True)
all_imgs = glob.glob(dir + '**/*.png')
print_class(all_imgs)
all_imgs.sort()

right_up_corn_logos = ['/Disk1/Dataset/TV_logo_data/全图版本/其他6/星空卫视HD.png',
                       '/Disk1/Dataset/TV_logo_data/全图版本/其他6/星空卫视.png']
# crop_save_logo(new_dir, all_imgs,right_up_corn_logos)
