CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --name unet_resvit_res_ls \
    --synthesize_model resvit_res_ls \
    --input fake_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask \
    --model_name unet \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \