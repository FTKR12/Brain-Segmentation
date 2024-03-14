CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --name unet_resvit_back \
    --synthesize_model resvit \
    --input fake_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask \
    --model_name unet \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \
    --background \