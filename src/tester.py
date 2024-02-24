import torch
import torch.nn as nn

from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import logging

logger = logging.getLogger("Brain Segmentation")

class Tester():
    def __init__(self, args, model, test_loader):
        self.args = args
        self.test_loader = test_loader

        self.model = model
        self.model.conv_final.add_module("activation", nn.Sigmoid())
        self.model.to(args.device)
        self.model.load_state_dict(torch.load(f'{self.args.output_dir}/best_model.pth'))

        # TODO: add metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
    
    def run(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                x, label = (batch["MRI"].to(self.args.device), batch["label"].to(self.args.device))
                # if inputs are multiple such as CT and MRI, remove following commentout
                # x = torch.cat([x1, x2], dim=1)

                out = sliding_window_inference(x, self.args.roi, self.args.sw_batch, self.model)
                out = [self.post_pred(i) for i in decollate_batch(out)]
                label = [self.post_label(i) for i in decollate_batch(label)]

                self.dice_metric(y_pred=out, y=label)
        
            dice_score = self.dice_metric.aggregate().item()
            logger.info(f'DICE SCORE: {dice_score}')

        return