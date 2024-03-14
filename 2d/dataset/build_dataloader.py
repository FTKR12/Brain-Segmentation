import os
import glob
import torch

from monai.data import (
    CacheDataset,
    DataLoader,
    ArrayDataset
)
from monai.transforms import(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)

def build_dataloader(args):

    # load data
    train_impaths, val_impaths, test_impaths = [], [], []
    train_segpaths, val_segpaths, test_segpaths = [], [], []
    train_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/train/{args.input}/*")
    val_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/val/{args.input}/*")
    test_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/test/{args.input}/*")
    for data_id in train_id:
        train_impaths.append(data_id)
        train_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    for data_id in val_id:
        val_impaths.append(data_id)
        val_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    for data_id in test_id:
        test_impaths.append(data_id)
        test_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    
    # augmentation
    train_imtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop((192, 192), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ])
    train_segtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop((192, 192), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ])
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    train_ds = ArrayDataset(train_impaths, train_imtrans, train_segpaths, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    val_ds = ArrayDataset(val_impaths, val_imtrans, val_segpaths, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
    
    test_ds = ArrayDataset(test_impaths, val_imtrans, test_segpaths, val_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader