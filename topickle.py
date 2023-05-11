# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/4/28 10:51
import pickle
import os
import csv
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms
import random
import cv2.cv2 as cv2

# fig_path = 'D:\\Pytorch\\Fusion_lowrank\\data\\original_figures'
# vib_path = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\Vibration_labeled.csv'
fig_path = '/Fusion_transformer/data/Sample366/figures366_record'
vib_path = '/Fusion_transformer/data/Sample366/figures366_record\\Vibration366_label.csv'
vib_feature_path = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\Vibration_feature_labeled_4200.csv'

# Image_list = []
IMAGE = {}
img_path_label = []

with open(os.path.join(fig_path, 'images.csv')) as f:
    reader = csv.reader(f)
    for row in reader:
        img, label = row
        img_path_label.append((img, int(label)))

for i in range(len(img_path_label)):
    image_path = img_path_label[i][0]
    # img = cv2.imread(image_path)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image = Image.open(image_path).convert('RGB')
    IMAGE.setdefault('IMAGE', {})[i] = image
    # IMAGE.setdefault('IMAGE', {})[i] = np.array(image).reshape((3264, 512, 1))
    # IMAGE.setdefault('IMAGE', {})[i] = trans(image)
    # IMAGE.setdefault('IMAGE', []).append(trans(image))
    IMAGE.setdefault('LABEL', {})[i] = img_path_label[i][1]

VIBRATION = {}

with open(vib_path) as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        value = row[0:12800]  # 12800/68
        vib_label = row[12800]
        vib_value = np.expand_dims(value, axis=-1)
        VIBRATION.setdefault('VIBRATION', {})[count] = np.array(vib_value, dtype='float64')
        # VIBRATION.setdefault('VIBRATION', {})[count] = (torch.tensor(np.array(vib_value, dtype='float64')))
        VIBRATION.setdefault('LABEL', {})[count] = int(vib_label)
        count += 1


# VIBRATION = {}
# with open(vib_path) as f:
#     reader = csv.reader(f)
#     count = 0
#     for row in reader:
#         value = row[0:12800]  # 12800/68
#         vib_label = row[12800]
#         value_lst = []
#         for h in range(30):
#             random_start = np.random.randint(low=0, high=(len(value) - 6400))
#             sample = value[random_start:random_start + 6400]
#             value_lst.append(value)
#         count += 1
#         VIBRATION.setdefault('VIBRATION', {})[count] = np.array(value_lst, dtype='float64')
#         VIBRATION.setdefault('LABEL', {})[count] = int(vib_label)

# VIBRATION = {}
# with open(vib_feature_path) as f:
#     reader = csv.reader(f)
#     count = 0
#     value_lst = []
#     for row in reader:
#         value = row[0:68]  # 12800/68
#         value_lst.append(value)
#     va = np.array(value_lst).reshape((140, 30, 68))
#     for i in va:
#         VIBRATION.setdefault('VIBRATION', {})[count] = np.array(i, dtype='float64')
#         count += 1

# with open(vib_path) as k:
#     reader = csv.reader(k)
#     count_ = 0
#     for row in reader:
#         vib_label = row[12800]
#         VIBRATION.setdefault('LABEL', {})[count_] = int(vib_label)
#         count_ += 1


def get_data(dic, keys):
    return [(dic[key]) for key in keys]
    # return np.array([(dic[key]) for key in keys])


# return [(key, dic[key]) for key in keys]


data_keys = list(IMAGE['IMAGE'].keys())
print(data_keys)
random.shuffle(data_keys)
train_keys = data_keys[:int(0.7 * len(data_keys))]
valid_keys = data_keys[int(0.7 * len(data_keys)):int(0.9 * len(data_keys))]
test_keys = data_keys[int(0.9 * len(data_keys)):]

IMAGE_Train = get_data(IMAGE['IMAGE'], train_keys)
IMAGE_Valid = get_data(IMAGE['IMAGE'], valid_keys)
IMAGE_Test = get_data(IMAGE['IMAGE'], test_keys)

# img_train_l = get_data(IMAGE['LABEL'], train_keys)
# img_valid_l = get_data(IMAGE['LABEL'], valid_keys)
# img_test_l = get_data(IMAGE['LABEL'], test_keys)

VIBRATION_Train = get_data(VIBRATION['VIBRATION'], train_keys)
VIBRATION_Valid = get_data(VIBRATION['VIBRATION'], valid_keys)
VIBRATION_Test = get_data(VIBRATION['VIBRATION'], test_keys)


# vib_train_l = get_data(VIBRATION['LABEL'], train_keys)
# vib_valid_l = get_data(VIBRATION['LABEL'], valid_keys)
# vib_test_l = get_data(VIBRATION['LABEL'], test_keys)
def label_process(labels):
    return np.array([[x] for x in labels])


LABEL_Train = label_process(get_data(IMAGE['LABEL'], train_keys))
LABEL_Valid = label_process(get_data(IMAGE['LABEL'], valid_keys))
LABEL_Test = label_process(get_data(IMAGE['LABEL'], test_keys))

# def dic_fusion(lst1, lst2):
#     fusion = [lst1, lst2]
#     Fusion_data = {}
#     for _ in fusion:
#         print(type(_))
#         for k in _:
#             for m, n in k.items():
#                 Fusion_data.setdefault(m, []).append(n)
#     return Fusion_data

Train = {'IMAGE': IMAGE_Train, 'VIBRATION': VIBRATION_Train, 'LABEL': LABEL_Train}
Valid = {'IMAGE': IMAGE_Valid, 'VIBRATION': VIBRATION_Valid, 'LABEL': LABEL_Valid}
Test = {'IMAGE': IMAGE_Test, 'VIBRATION': VIBRATION_Test, 'LABEL': LABEL_Test}

Rail_dataset = {'Train': Train, 'Valid': Valid, 'Test': Test}

file = open('D:\\Pytorch\\Fusion_lowrank\\data\\Rail_dataset_366.pkl', 'wb')
pickle.dump(Rail_dataset, file)
file.close()

'''
IMAGE = b'IMAGE'
VIBRATION = b'VIBRATION'
LABEL = b'LABEL'
TRAIN = b'Train'
VALID = b'Valid'
TEST = b'Test'

data_path = 'D:\\Pytorch\\Fusion_lowrank\\data\\'
Rail_data = pickle.load(open(data_path + "Rail_dataset.pkl", 'rb'))

Rail_train, Rail_valid, Rail_test = Rail_data[TRAIN], Rail_data[VALID], Rail_data[TEST]

train_img, train_vib, train_labels = Rail_train[IMAGE], Rail_train[VIBRATION], Rail_train[LABEL]
valid_img, valid_vib, valid_labels = Rail_valid[IMAGE], Rail_valid[VIBRATION], Rail_valid[LABEL]
test_img, test_vib, test_labels = Rail_test[IMAGE], Rail_test[VIBRATION], Rail_test[LABEL]
print('==============================')'''
