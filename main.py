import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from utils import *
from DME_deformable import DMENet
from config import DefaultConfig
import torchvision.transforms as transforms
import torch.cuda as torch_cuda
from data_process.DatasetConstructor import DatasetConstructor
import metrics
from PIL import Image
MAE = 10240000
MSE = 10240000
import time
# %matplotlib inline
# data_load
img_dir = "/home/zzn/part_A_final/train_data/images"
gt_dir = "/home/zzn/part_A_final/train_data/gt_map"
transform_a = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
gt_transform_a =  transforms.ToTensor()

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

dataset = DatasetConstructor(img_dir, gt_dir, 300, 20, transform_a, gt_transform_a)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
eval_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
args = DefaultConfig()

# model construct
net = DMENet().to(cuda_device)
gt_map_process_model = GroundTruthProcess(1, 1, 8).to(cuda_device) # to keep the same resolution with the prediction

# set optimizer and estimator
criterion = metrics.DMELoss().to(cuda_device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
ae_batch = metrics.AEBatch().to(cuda_device)
se_batch = metrics.SEBatch().to(cuda_device)
for epoch_index in range(args.max_epoch):
    dataset = dataset.train_model().shuffle()
    # train
    step = 0
    for train_img_index, train_img, train_gt, data_ptc in train_loader:
        # eval per 100 batch
        if step % 100 == 0:
            net.eval()
            dataset = dataset.eval_model().shuffle()
            loss_ = []
            MAE_ = []
            MSE_ = []

            rand_number = random.randint(0, 19)
            counter = 0

            for eval_img_index, eval_img, eval_gt, eval_data_ptc in eval_loader:
                if args.use_gpu:
                    # B
                    #                     eval_x = eval_img.view(-1, 3, 768, 1024).cuda()
                    #                     eval_y = eval_gt.view(-1, 1, 768, 1024).cuda()
                    # A
                    eval_x = eval_img.cuda()
                    eval_y = eval_gt.cuda()
                eval_prediction = net(eval_x)
                eval_groundtruth = gt_map_process_model(eval_y)
                # That’s because numpy doesn’t support CUDA,
                # so there’s no way to make it use GPU memory without a copy to CPU first.
                # Remember that .numpy() doesn’t do any copy,
                # but returns an array that uses the same memory as the tensor
                eval_loss = criterion(eval_prediction, eval_groundtruth).data.cpu().numpy()
                batch_ae = ae_batch(eval_prediction, eval_groundtruth).data.cpu().numpy()
                batch_se = se_batch(eval_prediction, eval_groundtruth).data.cpu().numpy()

                # random show 1 sample
                if rand_number == counter:
                    origin_image = Image.open(
                        "/home/zzn/part_A_final/train_data/images/IMG_" + str(eval_img_index.numpy()[0]) + ".jpg")
                    validate_pred_map = np.squeeze(eval_prediction.permute(0, 2, 3, 1).data.cpu().numpy())
                    validate_gt_map = np.squeeze(eval_groundtruth.permute(0, 2, 3, 1).data.cpu().numpy())

                    show(origin_image, validate_gt_map, validate_pred_map, eval_img_index.numpy()[0])
                    gt_counts = np.sum(validate_gt_map)
                    pred_counts = np.sum(validate_pred_map)
                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

                loss_.append(eval_loss)
                MAE_.append(batch_ae)
                MSE_.append(batch_se)
                counter += 1

            # calculate the validate loss, validate MAE and validate RMSE
            loss_ = np.reshape(loss_, [-1])
            MAE_ = np.reshape(MAE_, [-1])
            MSE_ = np.reshape(MSE_, [-1])

            validate_loss = np.mean(loss_)
            validate_MAE = np.mean(MAE_)
            validate_RMSE = np.sqrt(np.mean(MSE_))

            sys.stdout.write(
                'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, epoch_index + 1, validate_loss,
                                                                                  validate_MAE, validate_RMSE))
            sys.stdout.flush()

            # save model
            if MAE > validate_MAE:
                MAE = validate_MAE
                torch.save(net, args.mae_model_a)

            # save model
            if MSE > validate_RMSE:
                MSE = validate_RMSE
                torch.save(net, args.mse_model_a)

            # return train model
            net.train()
            dataset = dataset.train_model()

        net.train()
        optimizer.zero_grad()
        if args.use_gpu:
            # B
            #             x = train_img.view(-1, 3, 384, 512).cuda()
            #             y = train_gt.view(-1, 1, 384, 512).cuda()
            # A
            x = train_img.cuda()
            y = train_gt.cuda()
        else:
            print("only support gpu version")
            exit()
        prediction = net(x)
        groundtruth = gt_map_process_model(y)
        loss = criterion(prediction, groundtruth)
        loss.backward()
        optimizer.step()
        step += 1


