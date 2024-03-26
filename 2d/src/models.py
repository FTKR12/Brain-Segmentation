from monai.networks.nets import SegResNet, UNet, FlexibleUNet, SEResNext50, SwinUNETR
from monai.utils import UpsampleMode

# TODO: model args
def build_model(model_name, args):
    model = None

    # TODO: add models here
    if model_name == "segresnet":
        model = SegResNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
        )
    if model_name == "unet":
        if not "+" in args.input:
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        else:
            model = UNet(
                spatial_dims=2,
                in_channels=2,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
    if model_name == "flexible_unet":
        model = FlexibleUNet(
            in_channels=1,
            out_channels=1,
            backbone="efficientnet-b0",
            pretrained=True
        )
    if model_name == "swinunetr":
        model = SwinUNETR(
            img_size=args.roi,
            in_channels=1,
            out_channels=1,
            spatial_dims=2
        )

    assert model != None, "model is not supported. Add models in models.py yourself."

    return model