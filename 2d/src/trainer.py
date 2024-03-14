import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW

from monai.transforms import AsDiscrete, Compose, Activations
from monai.data import decollate_batch
from monai.networks import normal_init
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import logging

logger = logging.getLogger("Brain Segmentation")

class Trainer():
    # TODO: w&b or tensorboard for monitoring logs
    def __init__(self, args, model, train_loader, valid_loader):
        self.args = args

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model
        #self.model.apply(normal_init)
        #self.model.conv_final.add_module("activation", nn.Sigmoid())
        self.model = self.model.to(args.device)

        self.loss_func = DiceCELoss(sigmoid=True)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # TODO: add metrics
        self.dice_metric = DiceMetric(include_background=args.background, reduction="mean", get_not_nans=False)

        self.loss_logs, self.metric_logs = {'train': [], 'eval': []}, {'train': [], 'eval': []}
        self.best_metric, self.best_epoch = -1, -1

        #self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        #self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def run(self):
        for epoch in tqdm(range(self.args.epochs)):
            self.train_one_epoch()
            self.eval_one_epoch()
            self.save_checkpoint(epoch)
            self.print_logs(epoch)
        
        with open(f'{self.output_dir}/loss_logs.pkl', 'wb') as f:
            pickle.dump(self.loss_logs, f)
        with open(f'{self.output_dir}/metric_logs.pkl', 'wb') as f:
            pickle.dump(self.metric_logs, f)  
        
    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        for batch in self.train_loader:

            x, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
            
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_func(out, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= self.train_loader.__len__()
        self.loss_logs['train'].append(train_loss)

        return
    
    def eval_one_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_loader:
                x, label = batch[0].to(self.args.device), batch[1].to(self.args.device)
                out = sliding_window_inference(inputs=x, roi_size=self.args.roi, sw_batch_size=self.args.sw_batch, predictor=self.model)
                out = [self.post_pred(i) for i in decollate_batch(out)]
                self.dice_metric(y_pred=out, y=label)

            self.metric_logs['eval'].append(self.dice_metric.aggregate().item())
            self.dice_metric.reset()

        return

    def save_checkpoint(self, epoch):
        epoch_metric_value = self.metric_logs['eval'][epoch]
        if epoch_metric_value > self.best_metric:
            self.best_metric = epoch_metric_value
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "best_model.pth"))
            logger.info(f'model was saved, epoch {epoch} is the best.')

    def print_logs(self, epoch):
        logger.info(
            "\n"
            f"{'='*20}\n"
            f"Epoch: {epoch}\n"
            f"train loss: {self.loss_logs['train'][epoch]}\n"
            f"eval metric: {self.metric_logs['eval'][epoch]}\n"
            f"{'='*20}\n"
        )


