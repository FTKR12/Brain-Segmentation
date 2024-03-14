CUDA_VISIBLE_DEVICES=3 \
python train.py \
    --name swinunetr_real_mri \
    --synthesize_model resvit \
    --input real_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask \
    --model_name swinunetr \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \