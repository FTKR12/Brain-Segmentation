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

    train_images = sorted(glob.glob(os.path.join(args.dataset_dir, "train", "imagesTr", "*.nii.gz")))
    val_images = sorted(glob.glob(os.path.join(args.dataset_dir, "val", "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.dataset_dir, "train", "labelsTr", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(args.dataset_dir, "val", "labelsTr", "*.nii.gz")))
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear","bilinear", "nearest")),
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
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear","bilinear", "nearest")),
    ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_loader, val_loader