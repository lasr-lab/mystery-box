"""Model factory for DIGIT tactile classification."""

from __future__ import annotations
import timm
from typing import Any


SUPPORTED_MODELS = {
    "efficientnet_b0": "efficientnet_b0",
    "mobilevit_s": "mobilevit_s",
    "mobilevitv2_050": "mobilevitv2_050",
    "mobilevitv2_075": "mobilevitv2_075",
    "mobilevitv2_100": "mobilevitv2_100",
    "mobilevitv2_125": "mobilevitv2_125",
    "mobilevitv2_150": "mobilevitv2_150",
    "mobilevitv2_175": "mobilevitv2_175",
    "mobilevitv2_200": "mobilevitv2_200",
}


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def create_model(model_cfg: Any):
    """Create a pretrained single-image classifier from a Hydra model config."""
    model_name = _get_cfg_value(model_cfg, "name")
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}.")

    num_classes = _get_cfg_value(model_cfg, "num_classes")
    if num_classes is None:
        raise ValueError("Model config must define 'num_classes'.")
    return timm.create_model(
        SUPPORTED_MODELS[model_name],
        pretrained=bool(_get_cfg_value(model_cfg, "pretrained", True)),
        num_classes=int(num_classes),
        in_chans=int(_get_cfg_value(model_cfg, "input_channels", 3)),
    )
