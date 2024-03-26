import torch
import torch.nn as nn

from monai.transforms import AsDiscrete, Compose, Activations
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
        #self.model.conv_final.add_module("activation", nn.Sigmoid())
        self.model.to(args.device)
        self.model.load_state_dict(torch.load(f'{self.args.output_dir}/best_model.pth'))

        # TODO: add metrics
        self.dice_metric = DiceMetric(include_background=args.background, reduction="mean", get_not_nans=False)

        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    def run(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                x, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                # if inputs are multiple such as CT and MRI, remove following commentout
                # x = torch.cat([x1, x2], dim=1)

                out = sliding_window_inference(x, self.args.roi, self.args.sw_batch, self.model)
                out = [self.post_pred(i) for i in decollate_batch(out)]
                label = [i for i in decollate_batch(label)]

                self.dice_metric(y_pred=out, y=label)
        
            dice_score = self.dice_metric.aggregate().item()
            logger.info(f'DICE SCORE: {dice_score}')

        return