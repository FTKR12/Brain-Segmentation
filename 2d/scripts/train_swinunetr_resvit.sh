CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --name swinunetr_resvit \
    --synthesize_model resvit \
    --input fake_mri \
    --image_dir /mnt/strokeapp/Datasets/Seg_ctmri_2d \
    --mask_dir /mnt/strokeapp/Datasets/Seg_mask_2d \
    --model_name swinunetr \
    --epochs 300 \
    --train_batch_size 16 \
    --eval_batch_size 1 \