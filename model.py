import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes=2,
        encoder_name="resnet101",
        encoder_weights="imagenet",
        encoder_output_stride=16,
    ):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_output_stride=encoder_output_stride,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


def _normalize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model(
    checkpoint_path,
    num_classes=2,
    device=None,
    encoder_name="resnet101",
    encoder_weights="imagenet",
    encoder_output_stride=16,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_output_stride=encoder_output_stride,
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = _normalize_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device
