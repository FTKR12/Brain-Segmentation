# Brain Segmentation

This repogitory is implemented for braim MRI segmentation, which is based on [MONAI](https://docs.monai.io/en/stable/index.html).

## Enviroment
- Python >= 3.xx
- Pytorch >= x.xx

## Data
You should make folder structure as follows:
```
Brain-Segmentation/data/
  ├── train
    ├── 3 (id)
        ├── ct.nii
        ├── mri.nii
        ├── mask.nii
  ├── val
      ├── 5
        ├── ct.nii
        ├── mri.nii
        ├── mask.nii
  ├── test
      ├── 2
        ├── ct.nii
        ├── mri.nii
        ├── mask.nii
```

## Training
There are sample scripts for training in [scripts](scripts/).
For example,
```
bash scripts/train_segresnet.sh
```

## Testing
TBD