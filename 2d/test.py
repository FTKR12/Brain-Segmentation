import os

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.options import get_args
from dataset.build_dataloader import build_dataloader_single_input, build_dataloader_double_input
from src.tester import Tester
from src.models import build_model

def main(args):
    if not "+" in args.input:
        _, _, test_loader = build_dataloader_single_input(args)
    else:
        _, _, test_loader = build_dataloader_double_input(args)
    model = build_model(args.model_name, args)
    trainer = Tester(args, model, test_loader)
    trainer.run()
    
    return

if __name__ == '__main__':

    args = get_args()
    set_seed(args.seed)
    args.output_dir = f'{args.output_dir}/{args.name}'

    logger = setup_logger('Brain Segmentation', f'{args.output_dir}', is_Train=False)
    logger.info(str(args).replace(',','\n'))

    main(args)