import os.path
import random
from math import floor, ceil
from time import sleep
from tqdm import trange

import cv2
import glob
import numpy as np

from class_id import label_class


def truncated_normal(mean, stddev, minval, maxval):
    return np.clip(np.random.normal(mean, stddev), minval, maxval)

# for i in range(100):
#     random_w = truncated_normal(0.5, 0.18, 0, 1)
#     print(random_w)


#     print(np.random.randint(0, 2))


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


def fuse_image_multiple_logo(img_buffer, logos, masks,labels):
    '''
    :param input_img_path: 输入图像路径
    :param tag_img_path: 台标图像路径
    :return: imgs 融合后的图像,bboxs 相对的台标坐标
    '''
    target_size = 640
    bboxs = []
    ori_img = img_buffer
    assert len(logos) == len(masks)
    resize_flag = np.random.randint(0, 3)
    h, w, c = ori_img.shape
    if w <= target_size:  # 确保一张图的宽大于640 （OpenImage的宽都大于）
        ori_img = cv2.resize(ori_img, (int(target_size), int(target_size)))  # 该图只做一张训练使用的图
    else:
        if ori_img.shape[0] <= target_size:  # 确保一张图的高大于640 （OpenImage的高可能小于640）
            if resize_flag:  # 3/4 几率 resize
                ori_img = cv2.resize(ori_img, (int(target_size), int(target_size)))
            else:  # 1/4 几率 padding
                half_pad_h = (target_size - ori_img.shape[0]) / 2
                ori_img = cv2.copyMakeBorder(ori_img, floor(half_pad_h), ceil(half_pad_h), 0, 0, cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))
        else:  # 如果图的高大于640，crop
            if resize_flag:  # 大概率从最高处crop
                ori_img = ori_img[:target_size, :, :]
            else:
                ori_img = ori_img[-target_size:, :, :]
        ori_img = ori_img[:target_size, :target_size, :]
        # imgs.append(ori_img[:, :target_size, :target_size, ])
        # imgs.append(ori_img[:, -target_size:, -target_size:])
    h, w, c = ori_img.shape  # 640 x 640
    previous_h = floor(h * truncated_normal(0.103, 0.03, 0, 0.2))
    for logo_index, logo in enumerate(logos):
        # ------------------------得到随机的正太分布的logo相对大小--------------------
        w_target_ratio = truncated_normal(0.29, 0.1, 0.08, 0.5)  # 随机算法V3版本——0.29的宽度均值
        log_img = logos[logo_index]
        mask_img = masks[logo_index]

        small_h, small_w, small_c = log_img.shape
        target_w = round(w * w_target_ratio)  # 目标宽度
        target_h = round(small_h * (target_w) / small_w)  # 按比例缩放的
        if target_h > (h / 2):
            # print("The logo height is too high",target_w,target_h,h / 2)
            target_h = floor(h / 2)
            target_w = floor(small_w * (target_h) / small_h)
        log_img = cv2.resize(log_img, (int(target_w), int(target_h)))
        mask_img = cv2.resize(mask_img, (int(target_w), int(target_h)))

        # --------------中心坐标和高度、宽度------------------------
        small_h, small_w, small_c = log_img.shape  # logo的高、宽、通道数
        max_center_w = floor(640 - small_w / 2)
        max_center_h = floor(640 - small_h / 2)
        ratio_shift_w = truncated_normal(0.5, 0.18, 0, 1)  # 0.5x640的宽度位移（均值）
        # ratio_shift_h = truncated_normal(0.103, 0.03, 0, 0.2)  # 0.103X640=66 高度偏移均值
        w_shift = floor((max_center_w - small_w / 2) * ratio_shift_w)
        # h_shift = floor(h * ratio_shift_h)
        # np.linspace(0, 640, 4)   等距离采样，未使用
        log_center_w = ceil(small_w / 2) + w_shift
        log_center_h = ceil(small_h / 2) + previous_h
        if log_center_h + floor(small_h / 2) > max_center_h:  # logo堆叠的宽度已经比图像本身高，进行下一个图像的贴图
            break
        previous_h = log_center_h + ceil(small_h / 2) + 10  # 记录目前logo最下方的高度

        w_0, w_1, h_0, h_1 = log_center_w - ceil(small_w / 2), log_center_w + floor(small_w / 2), log_center_h \
                             - ceil(small_h / 2), log_center_h + floor(small_h / 2)
        # alpha 融合 fusion
        alpha = 0.08
        mask_img = mask_img[..., np.newaxis] / 255
        ori_img[h_0:h_1, w_0:w_1, :] = ori_img[h_0:h_1, w_0:w_1, :] * (1 - mask_img) + \
                                       ori_img[h_0:h_1, w_0:w_1, :] * mask_img * alpha + \
                                       log_img * mask_img * (1 - alpha)
        bbox = [labels[logo_index], log_center_w / w, log_center_h / h,
                small_w / w, small_h / h]
        bboxs.append(bbox)
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
                     logo_path='/Disk1/Dataset/电视台台标_筛掉后/', png_image='/Disk1/Dataset/V3_png'):
    logo_paths = glob.glob(logo_path + '*/*.jpg')
    png_paths = glob.glob(png_image + '/*.png')
    logo_paths += png_paths
    random.shuffle(logo_paths)  # 随机logo文件
    img_paths = glob.glob(img_path + '*.jpg')
    i = 0
    os.makedirs(dataset_path, exist_ok=True)
    bars = trange(len(img_paths), leave=True)
    for index in bars:
        if i > len(logo_paths):
            i = 0
        j = i + 4
        if j > len(logo_paths):
            j = len(logo_paths)
        logos = []
        masks = []
        labels=[]
        for logo_path in logo_paths[i:j]:
            if os.path.basename(logo_path).split('.')[1] == 'png':
                buffer = cv2.imread(logo_path,cv2.IMREAD_UNCHANGED)
                logo_img = buffer[:, :, 0:3]
                mask = buffer[:, :, 3]
                label=label_class[os.path.basename(logo_path).split('.')[0]]
            else:
                logo_img = cv2.imread(logo_path, 1)
                mask_path = os.path.join(os.path.dirname(logo_path),
                                         os.path.basename(logo_path).split('.')[0] + '_mask.png')
                mask = cv2.imread(mask_path, 0)
                label = label_class[os.path.basename(logo_path).split('.')[0]]
            logos.append(logo_img)
            masks.append(mask)
            labels.append(label)

        fused_img, bboxs = fuse_image_multiple_logo(cv2.imread(img_paths[index], 1),
                                                    logos=logos,
                                                    masks=masks,labels=labels)
        i += 4
        # fused_img = cv2.resize(fused_img, (640, 640))  # rezise为训练时的分辨率，加快训练速度
        cv2.imwrite(dataset_path + str(index) + '.jpg', fused_img)
        with open(dataset_path + str(index) + '.txt', 'w') as file:
            for bbox in bboxs:
                if bbox is None:
                    continue
                bbox = [str(b) for b in bbox]
                file.write(' '.join(bbox) + '\n')
                # print(bbox)


# if __name__ == '__main__':
# None
generate_dataset(dataset_path='/data/TVLogoDataset/TV_logo_data/train_01_V3/',
                 img_path='/data/TVLogoDataset/OpenImage/train_01/',
                 logo_path='/data/TVLogoDataset/电视台台标_筛掉后/',
			png_image='/data/TVLogoDataset/TV_logo_data/V3_png/')

# generate_dataset(dataset_path='/Disk1/Dataset/TV_logo_data/train_00_V3/',
#                  img_path='/Disk1/Dataset/OpenImage/train_00/',
#                  logo_path='/Disk1/Dataset/电视台台标_筛掉后/',png_image='/Disk1/Dataset/V3_png')
# a = glob.glob('/Disk1/Dataset/OpenImage/train_00/*')
# input_img_path=''
# getAllMask()
# None
# generate_class_mapping()

# logo_img_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CCTV1综合.jpg'
# mask_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CHC高清电影_mask.png'
# mask_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CCTV1综合_mask.png'

# ---------------- Test fuse_image_multiple_logo -----------------------
# input_img_path = '/Disk1/Dataset/OpenImage/train_00/ff892cac817c65e8.jpg'
# logo_img_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CHC高清电影.jpg'
# mask_img_path = '/Disk1/Dataset/电视台台标_筛掉后/CCTV/CHC高清电影_mask.png'
# img = cv2.imread(input_img_path, 1)
# logo=cv2.imread(logo_img_path, 1)
# logos = [logo,logo,logo,logo]
# mask=cv2.imread(mask_img_path, 0)
# masks = [mask,mask,mask,mask]
# fused_img, bbox = fuse_image_multiple_logo(img, logos, masks)
# cv2.imwrite('fused_test.jpg',fused_img)
