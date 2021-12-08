import glob
import os

import collections

label_class = {'东方卫视': 0, '四川卫视': 1, '云南卫视': 2, '内蒙古卫视': 3, 'CCTV1': 4, 'CCTV13': 5, '凤凰卫视': 6, '安徽卫视': 7, '陕西卫视': 8,
               'CCTV10': 9, '星空卫视HD': 10, '重庆卫视': 11, '山东卫视': 12, 'CCTV4': 13, '山西卫视': 14, '优漫卡通': 15, 'CCTV5': 16,
               'CCTV14': 17, 'CCTV17': 18, '青海卫视': 19, '浙江卫视': 20, 'CCTV9': 21, '黑龙江卫视': 22, '东南卫视': 23, 'CCTV3': 24,
               '广西卫视': 25, '宁夏卫视': 26, 'CCTV2': 27, 'CCTV5+': 28, '深圳卫视': 29, '河北卫视': 30, 'CCTV6': 31, '星空卫视': 32,
               'CCTV8': 33, '新疆卫视': 34, 'CCTV11': 35, '江苏卫视': 36, '天津卫视': 37, '吉林卫视': 38, '湖南卫视': 39, '北京卫视': 40,
               'CCTV15': 41, '甘肃卫视': 42, '广东卫视': 43, '湖北卫视': 44, '河南卫视': 45, '江西卫视': 46, '贵州卫视': 47, '凤凰卫视HD': 48,
               'CCTV16': 49, '金鹰卡通': 50, '辽宁卫视': 51, 'CCTV12': 52, '西藏卫视': 53, 'CCTV7': 54, '厦门卫视': 55, '海南卫视': 56}


def generate_class_mapping(dataset_path='/Disk1/Dataset/TV_logo_data/TV_logo_Crop/'):
    all_logo = glob.glob(dataset_path + '*.png')
    class_dict = {}
    for i, logo in enumerate(all_logo):
        class_dict[os.path.basename(logo.split('.')[0])] = i
    print(class_dict)
    label_name = [k for k in class_dict]
    print(label_name)


generate_class_mapping(dataset_path='/Disk1/Dataset/TV_logo_data/TV_logo_Crop/')
# print(label_name)
label_name =['东方卫视', '四川卫视', '云南卫视', '内蒙古卫视', 'CCTV1', 'CCTV13', '凤凰卫视', '安徽卫视', '陕西卫视', 'CCTV10', '星空卫视HD', '重庆卫视', '山东卫视', 'CCTV4', '山西卫视', '优漫卡通', 'CCTV5', 'CCTV14', 'CCTV17', '青海卫视', '浙江卫视', 'CCTV9', '黑龙江卫视', '东南卫视', 'CCTV3', '广西卫视', '宁夏卫视', 'CCTV2', 'CCTV5+', '深圳卫视', '河北卫视', 'CCTV6', '星空卫视', 'CCTV8', '新疆卫视', 'CCTV11', '江苏卫视', '天津卫视', '吉林卫视', '湖南卫视', '北京卫视', 'CCTV15', '甘肃卫视', '广东卫视', '湖北卫视', '河南卫视', '江西卫视', '贵州卫视', '凤凰卫视HD', 'CCTV16', '金鹰卡通', '辽宁卫视', 'CCTV12', '西藏卫视', 'CCTV7', '厦门卫视', '海南卫视']

