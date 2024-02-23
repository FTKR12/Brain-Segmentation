CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --name segresnet \
    --dataset_dir /mnt/strokeapp/Datasets/ctmri_mask \
    --mask_dir /mnt/strokeapp/Datasets/mask \
    --model segresnet \
    --epochs 100 \
    --train_batch_size 8 \
    --eval_batch_size 8 \