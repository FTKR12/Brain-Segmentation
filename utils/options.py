import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Brain Segmentation")
    ########## base options ##########
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--dataset_dir', default='dir/to/dataset/dir')
    parser.add_argument('--mask_dir', default='dir/to/mask/dir')

    ########## data options ##########
    parser.add_argument('--spatial_size', default=(96, 96, 96), type=tuple)

    ########## model options ##########
    parser.add_argument('--model' default='segresnet')
    # TODO: add mdoel args

    ########## training options ##########
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    ########## eval options ##########
    parser.add_argument('--eval_batch_size', default=4, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--roi', default=(128, 128, 96), type=tuple)
    parser.add_argument('--sw_batch', default=2, type=int)

    args = parser.parse_args()
    return args