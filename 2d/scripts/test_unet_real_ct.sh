CUDA_VISIBLE_DEVICES=0 \
python test.py \
    --name unet_real_ct \
    --synthesize_model resvit \
    --input real_ct \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name unet \
