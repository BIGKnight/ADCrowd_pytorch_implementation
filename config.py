class DefaultConfig(object):
    train_data_path = "/home/zzn/part_B_final/train_data/images_train.npy"
    test_data_path = "/home/zzn/part_B_final/test_data/images_test.npy"
    validate_data_path = "/home/zzn/part_B_final/train_data/images_validate.npy"

    train_gt_path = "/home/zzn/part_B_final/train_data/gt_train.npy"
    test_gt_path = "/home/zzn/part_B_final/test_data/gt_test.npy"
    validate_gt_path = "/home/zzn/part_B_final/train_data/gt_validate.npy"

    batch_size = 1
    use_gpu = True
    num_workers = 1
    validate_steps = 50
    max_epoch = 500
    lr = 1e-5
    lr_decay = 0.95
    weight_decay = 1e-4

    model_save_path = "/home/zzn/ADCrowd_pytorch/checkpoints/model.pkl"
