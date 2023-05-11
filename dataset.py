# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/4/25 9:40

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import csv
import glob
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = 'D:\\Mobilenet_CoordinateAttention_GAF\\Fusion_net\\The winter\\figures'


def default_loader(path):
    return Image.open(path).convert('RGB')


def load_image(data_path, mode='train'):
    class MyDataset(Dataset):
        def __init__(self, transform=None, target_transform=None, loader=default_loader):
            super(MyDataset, self).__init__()
            name2label = {}
            for name in sorted(os.listdir(os.path.join(data_path))):
                if not os.path.isdir(os.path.join(data_path, name)):
                    continue
                name2label[name] = len(name2label.keys())
            if not os.path.exists(os.path.join(data_path, 'images.csv')):
                images = []
                for key in name2label.keys():
                    images.extend(glob.glob(os.path.join(data_path, key, '*jpg')))
                print(images)
                with open(os.path.join(data_path, 'images.csv'), mode='w', newline='') as f:
                    writer = csv.writer(f)
                    for img in images:
                        name = img.split(os.sep)[-2]
                        label = name2label[name]
                        writer.writerow([img, label])
                    print('written into csv file:', 'images.csv')
            imgs = []
            with open(os.path.join(data_path, 'images.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    img, label = row
                    imgs.append((img, int(label)))

            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, item):
            name, label = self.imgs[item]
            img = self.loader(name)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_dataset = MyDataset(transform=transforms.ToTensor())

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    # train_dataset = train_dataset(transform=train_transform)
    print(train_dataset)


load_image(root)

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((128, 128)),
    transforms.ToTensor()
])

# full_dataset = MyDataset(transform=transforms.ToTensor())
# print(full_dataset)
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
# test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=4)
# print('num_of_trainData:', len(train_dataset))
# print('num_of_testData:', len(test_dataset))
