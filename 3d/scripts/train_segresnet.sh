CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --name segresnet \
    --synthesize_model resvit \
    --dataset_dir /mnt/strokeapp/Datasets/Seg_ctmri_3d \
    --model_name segresnet \
    --epochs 100 \
    --train_batch_size 1 \
    --eval_batch_size 1 \