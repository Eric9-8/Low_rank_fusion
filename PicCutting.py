# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/4/25 20:55
import os
import numpy as np
import cv2.cv2 as cv2

dataset_dir = 'D:/Mobilenet_CoordinateAttention_GAF/Fusion_net/The winter/figures/'
output_dir = '/Fusion_lowrank/utils/Pic/'
labels = ['normal', 'damage', 'switch', 'seam']


def cutting(input_path, out_path):
    img = cv2.imread(input_path)
    hei = img.shape[0]
    wid = img.shape[1]
    cropped = img[0:hei, 173:349]
    print('original:', img.shape, 'target:', cropped.shape)
    cv2.imwrite(out_path, cropped)


if __name__ == "__main__":
    for name in labels:
        original_path = os.path.join(dataset_dir, name)
        target_path = os.path.join(output_dir, name)

        image_file = [(os.path.join(original_path, x), os.path.join(target_path, x))
                      for x in os.listdir(original_path)]

        for path in image_file:
            cutting(path[0], path[1])
