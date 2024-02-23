import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Brain Segmentation")
    ########## base options ##########
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--dataset_dir', default='dir/to/dataset/dir')
    

    args = parser.parse_args()
    return args