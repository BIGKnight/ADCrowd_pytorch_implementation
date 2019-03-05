class DefaultConfig(object):
    train_data_path = "/home/zzn/part_B_final/train_data/images_train.npy"
    test_data_path = "/home/zzn/part_B_final/test_data/images_test.npy"
    validate_data_path = "/home/zzn/part_B_final/train_data/images_validate.npy"

    train_gt_path = "/home/zzn/part_B_final/train_data/gt_train.npy"
    test_gt_path = "/home/zzn/part_B_final/test_data/gt_test.npy"
    validate_gt_path = "/home/zzn/part_B_final/train_data/gt_validate.npy"

    train_dataset_A_path = "/home/zzn/part_A_final/train_data/images_train.npy"
    test_dataset_A_path = "/home/zzn/part_A_final/test_data/images_test.npy"
    validate_dataset_A_path = "/home/zzn/part_A_final/train_data/images_validate.npy"

    train_gt_A_path = "/home/zzn/part_A_final/train_data/gt_train.npy"
    test_gt_A_path = "/home/zzn/part_A_final/test_data/gt_test.npy"
    validate_gt_A_path = "/home/zzn/part_A_final/train_data/gt_validate.npy"


    batch_size = 1
    use_gpu = True
    num_workers = 1
    validate_steps = 500
    max_epoch = 500
    lr = 1e-5
    lr_decay = 0.95
    weight_decay = 1e-4

    mae_model_a = "/home/zzn/ADCrowd_pytorch/checkpoints/model_mae_a.pkl"
    mse_model_a = "/home/zzn/ADCrowd_pytorch/checkpoints/model_mae_a.pkl"
    mae_model_b = "/home/zzn/ADCrowd_pytorch/checkpoints/model_mae_b.pkl"
    mse_model_b = "/home/zzn/ADCrowd_pytorch/checkpoints/model_mse_b.pkl"
