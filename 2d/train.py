import os

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.options import get_args
from dataset.build_dataloader import build_dataloader_single_input, build_dataloader_double_input
from src.trainer import Trainer
from src.models import build_model
os.environ.get("MONAI_DATA_DIRECTORY")

def main(args, logger):
    
    if not "+" in args.input:
        train_loader, valid_loader, _ = build_dataloader_single_input(args)
    else:
        train_loader, valid_loader, _ = build_dataloader_double_input(args)
    model = build_model(args.model_name, args)
    logger.info(
        f"{'-'*100}\n"
        f"{model}\n"
        f"{'-'*100}\n"
    )
    logger.info(model)
    logger.info('-'*100)
    trainer = Trainer(args, model, train_loader, valid_loader)
    trainer.run()

    return

if __name__ == '__main__':

    args = get_args()
    set_seed(args.seed)
    os.makedirs(f"{args.fig_dir}/{args.model_name}/{args.input}", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/{args.name}', exist_ok=True)
    args.output_dir = f'{args.output_dir}/{args.name}'

    logger = setup_logger('Brain Segmentation', f'{args.output_dir}', is_Train=True)
    logger.info(str(args).replace(',','\n'))

    main(args, logger)