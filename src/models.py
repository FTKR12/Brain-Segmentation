from monai.networks.nets import SegResNet

# TODO: model args
def build_model(model_name):
    model = None

    # TODO: add models here
    if model_name == "segresnet":
        model = SegResNet()
    
    assert model != None, "model is not supported. Add models in models.py yourself."

    return model