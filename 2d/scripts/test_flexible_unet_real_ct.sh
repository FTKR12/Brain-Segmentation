CUDA_VISIBLE_DEVICES=2 \
python test.py \
    --name flexible_unet_real_ct \
    --synthesize_model resvit \
    --input real_ct \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name flexible_unet \