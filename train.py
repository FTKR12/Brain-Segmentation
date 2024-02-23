import os
import torch

from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
from monai.networks import normal_init
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.options import get_args
from dataset.build_dataloader import build_dataloader


def main(args. logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloader(args)

    model = SegResNet()
    model.apply(normal_init)
    model.conv_final.add_module("activation", torch.nn.Sigmoid())
    model = model.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print (model)

    max_epochs = 1000 #need 600 epochs to train a promising model
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 600)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs1,inputs2, labels = (
                batch_data["image1"].to(device),
                batch_data["image2"].to(device),
                batch_data["label"].to(device),
            )
            x = torch.cat([inputs1, inputs2], dim=1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs1,val_inputs2, val_labels = (
                        val_data["image1"].to(device),
                        val_data["image2"].to(device),
                        val_data["label"].to(device),
                    )
                    x = torch.cat([val_inputs1, val_inputs2], dim=1)
                    #roi_size = (192, 192, 128)
                    roi_size = (128, 128, 96)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(
                        x , roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

    return

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    logger = setup_logger('Brain Segmentation', f'{args.output_dir}')
    logger.info(str(args).replace(',','\n'))

    main(args, logger)