CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --name segresnet \
    --dataset_dir /mnt/strokeapp/Datasets/apis \
    --model_name segresnet \
    --epochs 100 \
    --train_batch_size 1 \
    --eval_batch_size 1 \