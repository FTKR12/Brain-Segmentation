CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --name segresnet_resvit_res_ls \
    --synthesize_model resvit_res_ls \
    --input fake_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name segresnet \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \