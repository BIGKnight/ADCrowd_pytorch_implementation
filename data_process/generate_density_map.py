import cv2
import numpy as np
import scipy
import scipy.io as scio
from PIL import Image


def get_density_map_gaussian(N, M, points, adaptive_kernel=False, fixed_value=15):
    density_map = np.zeros([N, M], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_kernel:
        # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances = tree.query(points, k=4)[0]

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_kernel:
                sigma = int(np.sum(distances[idx][1:4]) // 3 * 0.3)
            else:
                sigma = fixed_value
        else:
            sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = max(1, sigma)

        gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map


# 22, 37
if __name__ == "__main__":
    image_dir_path = "/home/zzn/part_B_final/train_data/images"
    ground_truth_dir_path = "/home/zzn/part_B_final/train_data/ground_truth"
    output_gt_dir = "/home/zzn/part_B_final/train_data/gt_map"
    for i in range(400):
        img_path = image_dir_path + "/IMG_" + str(i + 1) + ".jpg"
        gt_path = ground_truth_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
        img = Image.open(img_path)
        height = img.size[1]
        weight = img.size[0]
        points = scio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        gt = get_density_map_gaussian(height, weight, points, False, 5)
        gt = np.reshape(gt, [height, weight])  # transpose into w, h
        np.save(output_gt_dir + "/GT_IMG_" + str(i + 1), gt)
        print("complete!")
