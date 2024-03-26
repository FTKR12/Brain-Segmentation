CUDA_VISIBLE_DEVICES=3 \
python train.py \
    --name swinunetr_real_ct \
    --synthesize_model resvit \
    --input real_ct \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name swinunetr \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \