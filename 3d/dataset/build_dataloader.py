import os
import glob

from monai.data import (
    CacheDataset,
    DataLoader,
)
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
)

def build_dataloader(args):

    # load data
    train_paths, val_paths, test_paths = [], [], []
    train_id = glob.glob(f"{args.dataset_dir}/train/*")
    val_id = glob.glob(f"{args.dataset_dir}/val/*")
    test_id = glob.glob(f"{args.dataset_dir}/test/*")
    for data_id in train_id:
        train_paths.append({"image": f"{data_id}/output_train_{data_id[-3:]}_adc.nii", "label": f"{data_id}/output_train_{data_id[-3:]}_mask.nii"})
    for data_id in val_id:
        val_paths.append({"image": f"{data_id}/output_train_{data_id[-3:]}_adc.nii", "label": f"{data_id}/output_train_{data_id[-3:]}_mask.nii"})
    for data_id in test_id:
        test_paths.append({"image": f"{data_id}/output_train_{data_id[-3:]}_adc.nii", "label": f"{data_id}/output_train_{data_id[-3:]}_mask.nii"})

    # augmentation
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=18,
            image_threshold=0,
        ),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ])

    # maek dataset
    train_ds = CacheDataset(data=train_paths, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_paths, transform=val_transforms, cache_rate=1.0, num_workers=4)
    test_ds = CacheDataset(data=test_paths, transform=val_transforms, cache_rate=1.0, num_workers=4)
    
    # make dataloader
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, num_workers=4)

    return train_loader, val_loader, test_loader