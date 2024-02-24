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
    train_mri = glob.glob(os.path.join(args.dataset_dir, "train", "*.nii.gz"))
    val_mri = glob.glob(os.path.join(args.dataset_dir, "val", "*.nii.gz"))
    test_mri = glob.glob(os.path.join(args.dataset_dir, "test", "*.nii.gz"))
    train_labels = glob.glob(os.path.join(args.mask_dir, "train", "*.nii.gz"))
    val_labels = glob.glob(os.path.join(args.mask_dir, "val", "*.nii.gz"))
    test_labels = glob.glob(os.path.join(args.mask_dir, "test", "*.nii.gz"))
    train_paths = [{"mri": mri_path, "label": label_path} for mri_path, label_path in zip(train_mri, train_labels)]
    val_paths = [{"mri": mri_path, "label": label_path} for mri_path, label_path in zip(val_mri, val_labels)]
    test_paths = [{"mri": mri_path, "label": label_path} for mri_path, label_path in zip(test_mri, test_labels)]

    # augmentation
    train_transforms = Compose([
        LoadImaged(keys=["mri", "label"]),
        EnsureChannelFirstd(keys=["mri", "label"]),
        Orientationd(keys=["mri", "label"], axcodes="RAS"),
        Spacingd(keys=["mri", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["mri", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=18,
            image_threshold=0,
        ),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["mri", "label"]),
        EnsureChannelFirstd(keys=["mri", "label"]),
        Orientationd(keys=["mri", "label"], axcodes="RAS"),
        Spacingd(keys=["mri", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
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