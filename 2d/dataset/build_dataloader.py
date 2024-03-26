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
    Resized,
    RandBiasField,
    ThresholdIntensity,
    AdjustContrast,
    RandFlip,
    RandZoom
)

from dataset.original_dataset import OriginalArrayDataset

def build_dataloader_single_input(args):

    # load data
    train_impaths, val_impaths, test_impaths = [], [], []
    train_segpaths, val_segpaths, test_segpaths = [], [], []
    train_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/train/{args.input}/*")
    val_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/val/{args.input}/*")
    test_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/test/{args.input}/*")
    for data_id in train_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "").split("_")[0]) >= 60:
            #print(data_id)
            train_impaths.append(data_id)
            train_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    for data_id in val_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "").split("_")[0]) >= 60:
            val_impaths.append(data_id)
            val_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    for data_id in test_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "").split("_")[0]) >= 60:
            test_impaths.append(data_id)
            test_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{args.input}", ""))
    
    # augmentation
    train_imtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        #ThresholdIntensity(threshold=0.4, above=False, cval=0.9),
        #AdjustContrast(gamma=1.2),
        RandSpatialCrop(args.roi, random_size=False),
        #RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandZoom(prob=0.5, min_zoom=0.7, max_zoom=1.3),
        #RandBiasField(degree=2, prob=0.5, coeff_range=(0.2,0.5)),
    ])
    train_segtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop(args.roi, random_size=False),
        #RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandZoom(prob=0.5, min_zoom=0.7, max_zoom=1.3),
    ])
    #val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity(), ThresholdIntensity(threshold=0.4, above=False, cval=0.9), AdjustContrast(gamma=1.2)])
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    train_ds = ArrayDataset(train_impaths, train_imtrans, train_segpaths, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    val_ds = ArrayDataset(val_impaths, val_imtrans, val_segpaths, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    
    test_ds = ArrayDataset(test_impaths, val_imtrans, test_segpaths, val_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader

def build_dataloader_double_input(args):
    inputs = args.input.split("+")
    # load data
    train_ctpaths, val_ctpaths, test_ctpaths = [], [], []
    train_mripaths, val_mripaths, test_mripaths = [], [], []
    train_segpaths, val_segpaths, test_segpaths = [], [], []
    train_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/train/{inputs[0]}/*")
    val_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/val/{inputs[0]}/*")
    test_id = glob.glob(f"{args.image_dir}/{args.synthesize_model}/test/{inputs[0]}/*")
    for data_id in train_id:
        train_ctpaths.append(data_id)
        train_mripaths.append(data_id.replace(inputs[0], inputs[1]))
        train_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{inputs[0]}", ""))
    for data_id in val_id:
        val_ctpaths.append(data_id)
        val_mripaths.append(data_id.replace(inputs[0], inputs[1]))
        val_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{inputs[0]}", ""))
    for data_id in test_id:
        test_ctpaths.append(data_id)
        test_mripaths.append(data_id.replace(inputs[0], inputs[1]))
        test_segpaths.append(data_id.replace(args.image_dir, args.mask_dir).replace(f"/{args.synthesize_model}", "").replace(f"/{inputs[0]}", ""))
    
    # augmentation
    train_imtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop(args.roi, random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ])
    train_segtrans = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop(args.roi, random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ])
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    train_ds = OriginalArrayDataset(train_ctpaths, train_mripaths, train_imtrans, train_imtrans, train_segpaths, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    val_ds = OriginalArrayDataset(val_ctpaths, val_mripaths, val_imtrans, val_imtrans, val_segpaths, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    
    test_ds = OriginalArrayDataset(test_ctpaths, test_mripaths, val_imtrans, val_imtrans, test_segpaths, val_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader