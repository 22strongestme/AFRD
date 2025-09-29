import time
import numpy as np
import pandas as pd
import cv2
import os
import json
import random
import shutil
import argparse

def json2mask(json_path):
    with open(json_path, 'r') as f:
        context = f.read()
    js = json.loads(context)
    result = js.get('step_1').get('result')
    length = len(result)
    width = js.get('width')
    height = js.get('height')
    mask = np.zeros((height, width))
    for n in range(length):
        points = result[n]['pointList']
        p = []
        for i in range(len(points)):
            p.append([points[i]["x"], points[i]["y"]])
        points = p
        points = np.array(points, np.int32)
        mask = cv2.fillPoly(mask, [points], 255)
    return mask

def save_single_image(save_root, image_name, img, mask):
    imageDir = os.path.join(save_root, 'image')
    maskDir = os.path.join(save_root, 'mask')
    vizDir = os.path.join(save_root, 'viz')

    os.makedirs(imageDir, exist_ok=True)
    os.makedirs(maskDir, exist_ok=True)
    os.makedirs(vizDir, exist_ok=True)

    cv2.imwrite(os.path.join(imageDir, image_name), img)
    if np.max(mask) > 128:
        cv2.imwrite(os.path.join(maskDir, image_name), mask)

        vizImage = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        cv2.imwrite(os.path.join(vizDir, image_name), vizImage)

    return imageDir, maskDir


def reorganize_dataset(img_dir, mask_dir, save_root, class_name, test_ratio=0.2):
    img_list = os.listdir(img_dir)
    # # 正常和异常的图片划分
    img_label = []
    for img_path in img_list:
        if not os.path.exists(os.path.join(mask_dir, img_path)):
            j = False
        else:
            j = True
        img_label.append(j)
    normal_name = [img_list[i] for i in range(len(img_label)) if img_label[i] == False]
    abnormal_name = [img_list[i] for i in range(len(img_label)) if img_label[i] == True]
    # 打乱顺序
    random_normal_index = random.sample(range(len(normal_name)), len(normal_name))
    # 测试集数据名字
    test_normal_name = []
    test_abnormal_name = []

    test_normal_number = int(len(normal_name) * test_ratio)
    for i in range(test_normal_number):
        test_normal_name.append(normal_name[random_normal_index[i]])
    test_abnormal_name.extend(abnormal_name)

    # 训练集数据名字
    train_normal_name = []
    train_normal_number = int(len(normal_name) * (1 - test_ratio))
    bound = min(test_normal_number + train_normal_number, len(normal_name))
    for i in range(test_normal_number, bound):
        train_normal_name.append(normal_name[random_normal_index[i]])

    train_good_dir = os.path.join(save_root, class_name, 'train', 'good')
    test_good_dir = os.path.join(save_root, class_name, 'test', 'good')
    test_defect_dir = os.path.join(save_root, class_name, 'test', 'defect')
    gt_dir = os.path.join(save_root, class_name, 'ground_truth', 'defect')

    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # 测试集和训练集
    for train_path in train_normal_name:
        shutil.copy(os.path.join(img_dir, train_path), os.path.join(train_good_dir, train_path))
    for test_path in test_normal_name:
        shutil.copy(os.path.join(img_dir, test_path), os.path.join(test_good_dir, test_path))
    for test_path in test_abnormal_name:
        shutil.copy(os.path.join(img_dir, test_path), os.path.join(test_defect_dir, test_path))
    gt_list = os.listdir(mask_dir)
    for gt_path in gt_list:
        mask = cv2.imread(os.path.join(mask_dir, gt_path), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(gt_dir, f'{gt_path[:-4]}_mask.png'), mask)
        # shutil.copy(os.path.join(mask_dir, gt_path), os.path.join(gt_dir, gt_path))


def make_dataset(src_dir, save_all_images_root,
                save_dataset_root, test_spilt=0.2, use_nImages=20, **kwargs):

    # 确保每次生成的为新数据
    if os.path.exists(save_all_images_root):
        shutil.rmtree(save_all_images_root)
    os.makedirs(save_all_images_root, exist_ok=True)

    image_list = os.listdir(src_dir)
    image_list = [v for v in image_list if v.endswith('.bmp') or v.endswith('.png')]

    imageDirGlobal = None
    maskDirGlobal = None

    # we may not use all images to build the dataset, for speeding up
    actual_number = min(use_nImages, len(image_list))
    for index, image_path in enumerate(image_list):
        print(f'({index}): img: {image_path}')
        image = cv2.imread(os.path.join(src_dir, image_path), cv2.IMREAD_GRAYSCALE)
        json_path = image_path + '.json'
        json_path = os.path.join(src_dir, json_path)
        if os.path.exists(json_path):
            mask = json2mask(json_path)
        else:
            mask = np.zeros_like(image)

        if mask.max() < 128:
            if index > actual_number:
                isUsed = False
            else:
                isUsed = True
            print(f'normal: {isUsed}')
        else:
            isUsed = True
            print(f'abnormal: {isUsed}')

        if isUsed:
            mask = mask.astype(np.uint8)

            imageDir, maskDir = save_single_image(save_all_images_root, image_name=image_path, img=image, mask=mask)

            if maskDirGlobal == None:
                imageDirGlobal = imageDir
                maskDirGlobal = maskDir

    # 确保每次的为重新生成的数据
    if os.path.exists(save_dataset_root):
        shutil.rmtree(save_dataset_root)

    reorganize_dataset(img_dir=imageDirGlobal, mask_dir=maskDirGlobal,
                       save_root=save_dataset_root, class_name='cover_whole', test_ratio=test_spilt)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_args():
    parser = argparse.ArgumentParser(description='Make Dataset')
    parser.add_argument('--src_dir', type=str, default='../../datasets/angle1')
    parser.add_argument('--save_all_images_root',type=str,default='../../datasets/reorganized_angle1')
    parser.add_argument('--save_dataset_root',type=str, default='../../datasets/covercap_1010')
    parser.add_argument('--test_spilt', type=float, default=0.2)
    parser.add_argument('--use_nImages', type=int, default=30)
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    kwargs = vars(args)

    make_dataset(**kwargs)
