import matplotlib.pyplot as plt
import sys
from utils import GroundTruthProcess
from config import DefaultConfig
from data_process.DatasetConstructor import *
import metrics
# %matplotlib inline
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU
config = DefaultConfig()
img_dir = "/home/zzn/part_A_final/test_data/images"
gt_dir = "/home/zzn/part_A_final/test_data/gt_map"
transform_a = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
gt_transform_a =  transforms.ToTensor()

dataset = DatasetConstructor(img_dir, gt_dir, 182, 20, transform_a, gt_transform_a, True)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
mae_metrics = []
mse_metrics = []
net = torch.load("/home/zzn/ADCrowd_pytorch/checkpoints/2019_03_06.pkl").to(cuda_device)
net.eval()

gt_process_model = GroundTruthProcess(1, 1, 8).to(cuda_device)
ae_batch = metrics.AEBatch().to(cuda_device)
se_batch = metrics.SEBatch().to(cuda_device)
for real_index, test_img, test_gt, test_time_cost in test_loader:
    if config.use_gpu:
        predict_x = test_img.cuda()
        predict_gt = test_gt.cuda()
    predict_predict_map = net(predict_x)
    predict_gt_map = gt_process_model(predict_gt)
    batch_ae = ae_batch(predict_predict_map, predict_gt_map).data.cpu().numpy()
    batch_se = se_batch(predict_predict_map, predict_gt_map).data.cpu().numpy()
    mae_metrics.append(batch_ae)
    mse_metrics.append(batch_se)
    # to numpy
    numpy_predict_map = predict_predict_map.permute(0, 2, 3, 1).data.cpu().numpy()
    numpy_gt_map = predict_gt_map.permute(0, 2, 3, 1).data.cpu().numpy()

    # show current prediction
    figure, (origin, dm_gt, dm_pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(Image.open("/home/zzn/part_A_final/test_data/images/IMG_" + str(real_index.numpy()[0]) + ".jpg"))
    origin.set_title('Origin Image')
    dm_gt.imshow(np.squeeze(numpy_gt_map), cmap=plt.cm.jet)
    dm_gt.set_title('ground_truth')
    dm_pred.imshow(np.squeeze(numpy_predict_map), cmap=plt.cm.jet)
    dm_pred.set_title('prediction')
    plt.suptitle('The ' + str(real_index.numpy()[0]) + 'st images\'prediction')
    plt.show()
    sys.stdout.write('The grount truth crowd number is:{}, and the predicting number is:{}'.format(np.sum(numpy_gt_map),
                                                                                                   np.sum(
                                                                                                       numpy_predict_map)))
    sys.stdout.flush()
    mae_metrics = np.reshape(mae_metrics, [-1])
    mse_metrics = np.reshape(mse_metrics, [-1])
    MAE = np.mean(mae_metrics)
    MSE = np.sqrt(np.mean(mse_metrics))
    print('MAE:', MAE, 'MSE:', MSE)
    deformable = list(list(list(net.children())[1].children())[0].children())[0]
    offset = list(deformable.children())[0]
    mask = list(deformable.children())[1]
    print(offset, mask)