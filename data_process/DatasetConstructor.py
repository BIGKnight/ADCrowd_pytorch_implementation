from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt

crop_height = 600
crop_weight = 800


class DatasetConstructor(data.Dataset):
    default_transform = transforms.Compose([
        transforms.TenCrop((crop_height, crop_weight), vertical_flip=False),
        transforms.Lambda(
            lambda crops: [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2)(crop) for crop in crops
            ]
        ),
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

    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 generate_num,
                 if_train,
                 transformer=default_transform,
                 gt_transforms=gt_transform
                 ):
        self.length = generate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.transform = transformer
        self.gt_transform = gt_transforms
        self.train = if_train
        self.permulation = np.random.permutation(400)
        for i in range(self.length):
            img_name = '/IMG_' + str(self.permulation[i] + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(self.permulation[i] + 1) + ".npy"
            self.imgs.append([img_name, gt_map_name])

    def __getitem__(self, index):
        img, gt_map = self.imgs[index]
        img = Image.open(self.data_root + img).convert("RGB")
        gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map)))
        if self.train:
            if self.transform is not None:
                img = self.transform(img)
            if self.gt_transform is not None:
                gt_map = self.gt_transform(gt_map)

        else:
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])(img)
            gt_map = transforms.ToTensor()(gt_map)

        return self.permulation[index] + 1, img, gt_map

    def __len__(self):
        return self.length



