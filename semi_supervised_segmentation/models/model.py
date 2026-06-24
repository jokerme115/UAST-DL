import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .. import config as config_module


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=None):
        super(DeepLabV3Plus, self).__init__()
        if num_classes is None:
            num_classes = getattr(config_module, 'NUM_CLASSES', 2)
        encoder_name = getattr(config_module, 'ENCODER_NAME', 'resnet101')
        encoder_weights = getattr(config_module, 'ENCODER_WEIGHTS', 'imagenet')
        encoder_output_stride = getattr(config_module, 'ENCODER_OUTPUT_STRIDE', 16)
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_output_stride=encoder_output_stride,
            classes=num_classes,
            activation=None
        )

    def forward(self, x):
        return self.model(x)
