from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import random
import time

crop_height = 384
crop_weight = 512

default_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.TenCrop((crop_height, crop_weight), vertical_flip=False),
        transforms.Lambda(
            lambda crops: [
                transforms.ToTensor()(crop) for crop in crops
            ]
        ),
        transforms.Lambda(
            lambda crops: [
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops
            ]
        ),
        transforms.Lambda(lambda crops: torch.stack(crops))
    ])

gt_transform = transforms.Compose([
        transforms.TenCrop((crop_height, crop_weight), vertical_flip=False),
        transforms.Lambda(
            lambda crops: [
                transforms.ToTensor()(crop) for crop in crops
            ]
        ),
        transforms.Lambda(lambda crops: torch.stack(crops))
    ])


class DatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 train_num,
                 validate_num,
                 transformer,
                 gt_transformer,
                 if_train=True
                 ):
        self.train_num = train_num
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.transform = transformer
        self.gt_transform = gt_transformer
        self.train = if_train
        self.train_permulation = np.random.permutation(self.train_num)
        self.eval_permulation = random.sample(range(0, self.train_num - 1),  self.validate_num)
        for i in range(self.train_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):

        start = time.time()
        if self.train:
            img, gt_map = self.imgs[self.train_permulation[index]]
            if self.transform is not None:
                img = self.transform(img)
            if self.gt_transform is not None:
                gt_map = self.gt_transform(gt_map)
            end = time.time()
            return self.train_permulation[index] + 1, img, gt_map, (end - start)
        else:
            img, gt_map = self.imgs[self.eval_permulation[index]]
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])(img)
            gt_map = transforms.ToTensor()(gt_map)
            end = time.time()
            return self.eval_permulation[index] + 1, img, gt_map, (end - start)

    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return self.validate_num

    def shuffle(self):
        if self.train:
            self.train_permulation = np.random.permutation(self.train_num)
        else:
            self.eval_permulation = random.sample(range(0, self.train_num - 1),  self.validate_num)
        return self

    def eval_model(self):
        self.train = False
        return self

    def train_model(self):
        self.train = True
        return self
