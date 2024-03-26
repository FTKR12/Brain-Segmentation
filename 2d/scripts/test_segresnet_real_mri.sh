CUDA_VISIBLE_DEVICES=2 \
python test.py \
    --name segresnet_real_mri \
    --synthesize_model resvit \
    --input real_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name segresnet \