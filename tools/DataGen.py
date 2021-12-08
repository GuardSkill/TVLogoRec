import os.path
from math import floor, ceil
from time import sleep
from tqdm import trange

import cv2
import glob
import numpy as np

from class_id import label_class

def truncated_normal(mean, stddev, minval, maxval):
    return np.clip(np.random.normal(mean, stddev), minval, maxval)


#
# for i in range(100):
# print(truncated_normal(0.5, 0.18, 0, 1))
# print(np.random.randint(0, 2))


def fuse_image(input_img_path, logo_img_path, mask_path):
    '''
    :param input_img_path: 输入图像路径
    :param tag_img_path: 台标图像路径
    :return: img 融合后的图像,bbox 表示台标的,class_id 类别
    '''

    # w_max_ratio = 3 / 10  # logo的最大宽度占屏幕        一般-BRTV: 200/1000 CCTV 0.1315 #  w [0.06-0.26]
    # h_max_ratio = 2 / 10  # logo占屏幕的最大高度     最大180/1000     CCTV 0.125          # h [-0.25]
    # ------------------------得到随机的正太分布的logo相对大小--------------------
    random_w = truncated_normal(0, 0.2, -0.65, 0.65)
    w_target_ratio = 0.1315 + 0.1315 * random_w  # V1 版本_0.1315的宽
    ori_img = cv2.imread(input_img_path, 1)
    log_img = cv2.imread(logo_img_path, 1)
    mask_img = cv2.imread(mask_path, 0)

    h, w, c = ori_img.shape
    small_h, small_w, small_c = log_img.shape
    target_w = round(w * w_target_ratio)  # 目标宽度
    target_h = round(small_h * (target_w) / small_w)  # 按比例缩放的
    if target_h > (h / 2):
        target_h = floor(h / 2)
        target_w = floor(small_w * (target_h) / small_h)
    log_img = cv2.resize(log_img, (int(target_w), int(target_h)))
    mask_img = cv2.resize(mask_img, (int(target_w), int(target_h)))
    # --------------中心坐标和高度、宽度------------------------
    small_h, small_w, small_c = log_img.shape
    max_center_w = floor(w / 2 - small_w / 2)
    max_center_h = floor(h / 2 - small_h / 2)
    if max_center_w - ceil(small_w / 2) < 0 or max_center_h - ceil(small_h / 2) < 0:
        print("WARNING: None Appropriate Position for Logo, don't add logo!")
        return ori_img, [' ']
    ratio_shift_w = truncated_normal(0.5, 0.18, 0, 1)
    ratio_shift_h = truncated_normal(0.5, 0.18, 0, 1)
    w_shift = floor((max_center_w - small_w / 2) * ratio_shift_w) + floor(w / 2 * np.random.randint(0, 2))
    h_shift = floor((max_center_h - small_h / 2) * ratio_shift_h) + floor(h / 2 * np.random.randint(0, 2))
    log_center_w = ceil(small_w / 2) + w_shift
    log_center_h = ceil(small_h / 2) + h_shift

    w_0, w_1, h_0, h_1 = log_center_w - ceil(small_w / 2), log_center_w + floor(small_w / 2), log_center_h \
                         - ceil(small_h / 2), log_center_h + floor(small_h / 2)
    # fusion
    alpha = 0.75
    mask_img = mask_img[..., np.newaxis] / 255
    ori_img[h_0:h_1, w_0:w_1, :] = ori_img[h_0:h_1, w_0:w_1, :] * (1 - mask_img) + log_img * mask_img * alpha
    bbox = [log_center_w / w, log_center_h / h, small_w / w, small_h / h]
    return ori_img, bbox


def fuse_image_multiple_logo(input_img_path, logo_img_paths):
    '''
    :param input_img_path: 输入图像路径
    :param tag_img_path: 台标图像路径
    :return: img 融合后的图像,bbox 表示台标的,class_id 类别
    '''
    logo_index = 0
    bboxs = []
    ori_img = cv2.imread(input_img_path, 1)
    for logo_img_path in logo_img_paths:
        # w_max_ratio = 3 / 10  # logo的最大宽度占屏幕        一般-BRTV: 200/1000 CCTV 0.1315 #  w [0.06-0.26]
        # h_max_ratio = 2 / 10  # logo占屏幕的最大高度     最大180/1000     CCTV 0.125          # h [-0.25]
        # ------------------------得到随机的正太分布的logo相对大小--------------------
        random_w = truncated_normal(0, 0.2, -0.65, 0.65)
        w_target_ratio = 0.2 + 0.2 * random_w  # V1 版本_0.1315的宽

        log_img = cv2.imread(logo_img_path, 1)
        mask_path = os.path.join(os.path.dirname(logo_img_path),
                                 os.path.basename(logo_img_path).split('.')[0] + '_mask.png')
        mask_img = cv2.imread(mask_path, 0)

        h, w, c = ori_img.shape
        small_h, small_w, small_c = log_img.shape
        target_w = round(w * w_target_ratio)  # 目标宽度
        target_h = round(small_h * (target_w) / small_w)  # 按比例缩放的
        if target_h > (h / 2):
            target_h = floor(h / 2)
            target_w = floor(small_w * (target_h) / small_h)
        log_img = cv2.resize(log_img, (int(target_w), int(target_h)))
        mask_img = cv2.resize(mask_img, (int(target_w), int(target_h)))

        # --------------中心坐标和高度、宽度------------------------
        small_h, small_w, small_c = log_img.shape  # logo的高、宽、通道数
        max_center_w = floor(w / 2 - small_w / 2)
        max_center_h = floor(h / 2 - small_h / 2)
        if max_center_w - ceil(small_w / 2) < 0 or max_center_h - ceil(small_h / 2) < 0:
            # 1/2的原图宽度不能容纳下resize后的logo
            print("WARNING: None Appropriate Position for Logo, don't add logo!")
            return ori_img, bboxs
        ratio_shift_w = truncated_normal(0.5, 0.18, 0, 1)
        ratio_shift_h = truncated_normal(0.5, 0.18, 0, 1)
        w_shift = floor((max_center_w - small_w / 2) * ratio_shift_w) + floor(w / 2 * (logo_index // 2))
        h_shift = floor((max_center_h - small_h / 2) * ratio_shift_h) + floor(h / 2 * (logo_index % 2))
        log_center_w = ceil(small_w / 2) + w_shift
        log_center_h = ceil(small_h / 2) + h_shift

        w_0, w_1, h_0, h_1 = log_center_w - ceil(small_w / 2), log_center_w + floor(small_w / 2), log_center_h \
                             - ceil(small_h / 2), log_center_h + floor(small_h / 2)
        # fusion
        alpha = 0.75
        mask_img = mask_img[..., np.newaxis] / 255
        ori_img[h_0:h_1, w_0:w_1, :] = ori_img[h_0:h_1, w_0:w_1, :] * (1 - mask_img) + log_img * mask_img * alpha
        bbox = [label_class[os.path.basename(logo_img_path).split('.')[0]], log_center_w / w, log_center_h / h,
                small_w / w, small_h / h]
        bboxs.append(bbox)
        logo_index += 1
    return ori_img, bboxs


def getMaskFromLogo(logo_img_path, close_mask=0):
    '''
    :param tag_img_path: 台标图像路径
    :return: mask 对应mask
    '''
    logo_img = cv2.imread(logo_img_path)
    # cv2.imwrite('orgin.jpg', logo_img)
    l = len(logo_img.shape)
    if l == 3:  # 是彩色图
        row, col, channel = logo_img.shape
        if channel == 3:
            gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(logo_img, cv2.COLOR_BGRA2GRAY)
    else:  # 是灰度图
        gray = logo_img
    # retval, mask = cv2.threshold(gray, 242, 255,cv2.THRESH_BINARY)
    # 自适应的二值化方法
    mask = cv2.adaptiveThreshold(src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType=cv2.THRESH_BINARY_INV, blockSize=101 * 101, C=12)

    if close_mask > 0:
        kernel = np.ones((close_mask, close_mask), dtype='uint8')
        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 形态学方法
        contours, hierarchy = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ### cv2.RETR_EXTERNAL 只检测外轮廓   cv2.RETR_TREE            建立一个等级树结构的轮廓。
        # cv2.drawContours(logo_img, contours, -1, (255, 0, 255), 2)
        new_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(new_mask, contours, -1, (255), -1)  # 重新将轮廓内涂白2
        mask = new_mask
    return mask


def get_all_mask(logo_path='/Disk1/Dataset/电视台台标_筛掉后/'):
    '''
    为所有台标生成mask图
    '''
    logo_paths = glob.glob(logo_path + '*/*.jpg')
    for logo_path in logo_paths:
        mask = getMaskFromLogo(logo_path)
        mask_path = os.path.join(os.path.dirname(logo_path), os.path.basename(logo_path).split('.')[0] + '_mask.png')
        cv2.imwrite(mask_path, mask)


def generate_dataset(dataset_path='/Disk1/Dataset/TV_logo_data/train_00/',
                     img_path='/Disk1/Dataset/OpenImage/train_00/',
                     logo_path='/Disk1/Dataset/电视台台标_筛掉后/'):
    logo_paths = glob.glob(logo_path + '*/*.jpg')
    img_paths = glob.glob(img_path + '*.jpg')
    i = 0
    index = 0
    os.makedirs(dataset_path, exist_ok=True)
    bars = trange(len(img_paths), leave=True)
    for index in bars:
        if i > len(logo_paths):
            i = 0
        j = i + 4
        if j > len(logo_paths):
            j = len(logo_paths)
        fused_img, bboxs = fuse_image_multiple_logo(img_paths[index], logo_paths[i:j])
        i += 4
        fused_img = cv2.resize(fused_img, (640, 640))        # rezise为训练时的分辨率，加快训练速度
        cv2.imwrite(dataset_path + str(index) + '.jpg', fused_img)
        with open(dataset_path + str(index) + '.txt', 'w') as file:
                for bbox in bboxs:
                    if bbox is None:
                        continue
                    bbox = [str(b) for b in bbox]
                    file.write(' '.join(bbox) + '\n')
                    # print(bbox)


if __name__ == '__main__':
    generate_dataset(dataset_path='/data/TVLogoDataset/TV_logo_data/train_00_V2/',
                     img_path='/data/TVLogoDataset/OpenImage/train_00/',
                     logo_path='/data/TVLogoDataset/电视台台标_筛掉后/')
# a=glob.glob('/Disk1/Dataset/OpenImage/train_00/*')
# input_img_path=''
# getAllMask()
# None
# generate_class_mapping()
# logo_img_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CHC高清电影.jpg'
# logo_img_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CCTV1综合.jpg'
# mask_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CHC高清电影_mask.png'
# mask_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CCTV1综合_mask.png'
# input_img_path = '/Disk1/Dataset/OpenImage/train_00/ff892cac817c65e8.jpg'
# fused_img, bbox = fuseImage(input_img_path, logo_img_path, mask_path)
