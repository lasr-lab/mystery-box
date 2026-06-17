"""Checkpoint-backed inference utilities for the DIGIT tactile demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.ML.data.digit_fabrics import build_transforms
from src.ML.models import create_model


@dataclass(frozen=True)
class Prediction:
    """Single-frame classification result."""

    label: str
    index: int
    probabilities: list[float]


class DemoClassifier:
    """Load a training checkpoint and classify one OpenCV BGR frame at a time."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_cfg: Any,
        data_cfg: Any,
        device: str | torch.device = "auto",
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self._checkpoint_path}")

        self._device = _select_device(device)
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}.")

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"Checkpoint is missing required key 'model_state_dict': {self._checkpoint_path}")

        self._class_names = _class_names_from_checkpoint(checkpoint, self._checkpoint_path)
        class_count = len(self._class_names)

        checkpoint_config = _to_container(checkpoint.get("config"))
        effective_data_cfg = _config_section(checkpoint_config, "data") or _to_container(data_cfg)
        if effective_data_cfg is None:
            raise ValueError("Data config is required because checkpoint does not contain config.data.")
        _validate_data_class_count(effective_data_cfg, self._class_names)

        effective_model_cfg = _config_section(checkpoint_config, "model") or _to_container(model_cfg)
        if effective_model_cfg is None:
            raise ValueError("Model config is required because checkpoint does not contain config.model.")
        effective_model_cfg = _prepare_model_cfg(
            effective_model_cfg,
            class_count=class_count,
            data_cfg=effective_data_cfg,
            checkpoint_model_name=checkpoint.get("model_name"),
        )

        self._transform = build_transforms(effective_data_cfg, train=False)
        self._model = create_model(effective_model_cfg)
        try:
            self._model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load model_state_dict from {self._checkpoint_path}: {exc}") from exc
        self._model.to(self._device)
        self._model.eval()

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(self, frame_bgr: Any) -> Prediction:
        """Classify one OpenCV BGR frame."""

        image = _bgr_frame_to_rgb_image(frame_bgr)
        tensor = self._transform(image)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected transform to return a torch.Tensor, got {type(tensor).__name__}.")

        batch = tensor.unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(batch)
            if not torch.is_tensor(logits):
                raise TypeError(f"Expected model to return a torch.Tensor, got {type(logits).__name__}.")
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise RuntimeError(f"Expected logits with shape [1, num_classes], got {tuple(logits.shape)}.")
            if logits.shape[1] != len(self._class_names):
                raise RuntimeError(
                    "Model output class count mismatch: "
                    f"logits have {logits.shape[1]} classes but checkpoint defines {len(self._class_names)} labels."
                )

            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

        index = int(torch.argmax(probabilities).item())
        return Prediction(
            label=self._class_names[index],
            index=index,
            probabilities=[float(value) for value in probabilities.tolist()],
        )


def _select_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        selected = device
    else:
        requested = str(device).strip().lower()
        if requested == "auto":
            selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            selected = torch.device(requested)

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but CUDA is not available.")
    return selected


def _class_names_from_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> list[str]:
    raw_class_names = checkpoint.get("class_names")
    if raw_class_names is None:
        raise KeyError(f"Checkpoint is missing required key 'class_names': {path}")
    if isinstance(raw_class_names, (str, bytes)) or not isinstance(raw_class_names, Sequence):
        raise ValueError("Checkpoint 'class_names' must be a sequence of class label strings.")

    class_names = []
    for index, class_name in enumerate(raw_class_names):
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                "Checkpoint 'class_names' entries must be non-empty strings: "
                f"entry {index} is {class_name!r} ({type(class_name).__name__})."
            )
        class_names.append(class_name)
    if not class_names:
        raise ValueError("Checkpoint 'class_names' must contain at least one class.")
    return class_names


def _to_container(cfg: Any) -> Any:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return {key: _to_container(value) for key, value in cfg.items()}
    if isinstance(cfg, list):
        return [_to_container(value) for value in cfg]
    if isinstance(cfg, tuple):
        return tuple(_to_container(value) for value in cfg)
    return cfg


def _config_section(cfg: Any, key: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    return None


def _prepare_model_cfg(
    model_cfg: Any,
    *,
    class_count: int,
    data_cfg: Any,
    checkpoint_model_name: Any,
) -> dict[str, Any]:
    if isinstance(model_cfg, Mapping):
        prepared = dict(model_cfg)
    elif hasattr(model_cfg, "__dict__"):
        prepared = dict(vars(model_cfg))
    else:
        raise ValueError(f"Model config must be a mapping or config object, got {type(model_cfg).__name__}.")

    configured_model_name = prepared.get("name")
    has_configured_model_name = not _is_missing_or_interpolation(configured_model_name)
    has_checkpoint_model_name = not _is_missing_or_interpolation(checkpoint_model_name)
    if has_configured_model_name and has_checkpoint_model_name:
        if configured_model_name != checkpoint_model_name:
            raise ValueError(
                "Model config name mismatch: "
                f"effective model config name={configured_model_name!r} but checkpoint model_name={checkpoint_model_name!r}."
            )
    elif not has_configured_model_name and has_checkpoint_model_name:
        prepared["name"] = str(checkpoint_model_name)

    configured_num_classes = prepared.get("num_classes")
    if _is_missing_or_interpolation(configured_num_classes):
        prepared["num_classes"] = class_count
    elif int(configured_num_classes) != class_count:
        raise ValueError(
            "Model config class count mismatch: "
            f"num_classes={configured_num_classes} but checkpoint defines {class_count} class names."
        )

    input_channels = prepared.get("input_channels")
    data_input_channels = _nested_get(data_cfg, "input", "color_channels")
    if _is_missing_or_interpolation(input_channels) and data_input_channels is not None:
        prepared["input_channels"] = int(data_input_channels)

    # The checkpoint state dict supplies all learned weights; avoid external pretrained downloads at demo time.
    prepared["pretrained"] = False
    return prepared


def _validate_data_class_count(data_cfg: Any, class_count: int | Sequence[str]) -> None:
    checkpoint_class_names: list[str] | None
    if isinstance(class_count, int):
        checkpoint_class_names = None
        expected_class_count = class_count
    else:
        if isinstance(class_count, (str, bytes)) or not isinstance(class_count, Sequence):
            raise ValueError("Checkpoint class names must be a sequence of class label strings.")
        checkpoint_class_names = list(class_count)
        expected_class_count = len(checkpoint_class_names)

    configured_num_classes = _get(data_cfg, "num_classes")
    if (
        not _is_missing_or_interpolation(configured_num_classes)
        and int(configured_num_classes) != expected_class_count
    ):
        raise ValueError(
            "Data config class count mismatch: "
            f"num_classes={configured_num_classes} but checkpoint defines {expected_class_count} class names."
        )

    configured_classes = _get(data_cfg, "classes")
    if configured_classes is None or _is_missing_or_interpolation(configured_classes):
        return
    if isinstance(configured_classes, (str, bytes)) or not isinstance(configured_classes, Sequence):
        raise ValueError("Data config 'classes' must be a sequence when provided.")
    configured_class_names = []
    for index, class_name in enumerate(configured_classes):
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                "Data config 'classes' entries must be non-empty strings: "
                f"entry {index} is {class_name!r} ({type(class_name).__name__})."
            )
        configured_class_names.append(class_name)

    if len(configured_class_names) != expected_class_count:
        raise ValueError(
            "Data config class count mismatch: "
            f"{len(configured_class_names)} configured classes but checkpoint defines "
            f"{expected_class_count} class names."
        )
    if checkpoint_class_names is not None and configured_class_names != checkpoint_class_names:
        raise ValueError(
            "Data config class names/order mismatch: "
            f"configured classes={configured_class_names!r} but checkpoint class_names={checkpoint_class_names!r}."
        )


def _bgr_frame_to_rgb_image(frame_bgr: Any) -> Image.Image:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected OpenCV BGR frame with shape [height, width, 3], got {frame.shape}.")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected OpenCV BGR frame with dtype uint8, got {frame.dtype}.")

    frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
    return Image.fromarray(frame_rgb)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(cfg: Any, *keys: str) -> Any:
    current = cfg
    for key in keys:
        current = _get(current, key)
        if current is None:
            return None
    return current


def _is_missing_or_interpolation(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.startswith("${"))


__all__ = ["DemoClassifier", "Prediction"]
