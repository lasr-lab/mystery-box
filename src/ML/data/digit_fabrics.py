"""Dataset helpers for single-image DIGIT fabric classification."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class DigitFabricsDataset(Dataset):
    """Small image-file dataset for DIGIT tactile fabric frames."""

    def __init__(
        self,
        samples: Sequence[ImageSample],
        class_names: Sequence[str],
        transform: Optional[Any] = None,
        horizontal_flip_p: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.classes = list(class_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.transform = transform
        self.horizontal_flip_p = horizontal_flip_p
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as raw_image:
            image = raw_image.convert("RGB")
        if self._should_flip(sample):
            image = ImageOps.mirror(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    def _should_flip(self, sample: ImageSample) -> bool:
        if self.horizontal_flip_p <= 0.0:
            return False
        if self.horizontal_flip_p >= 1.0:
            return True

        key = f"{self.seed}:{sample.label}:{sample.path.name}".encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), byteorder="big")
        return value / 2**64 < self.horizontal_flip_p


def create_datasets(
    cfg: Any,
    raw_data_dir: Union[str, Path],
    seed: Optional[int] = None,
) -> tuple[DigitFabricsDataset, DigitFabricsDataset, DigitFabricsDataset]:
    """Create deterministic train/val/test datasets from class folders.

    Args:
        cfg: Data config node, or a root config with a ``data`` section.
        raw_data_dir: Path to ``data/raw``.
        seed: Fixed seed used for class-balanced sampling, splitting, and
            deterministic train-time augmentation. Defaults to ``cfg.seed``.
    """

    data_cfg = _data_cfg(cfg)
    class_names = list(_get(data_cfg, "classes"))
    configured_num_classes = _get(data_cfg, "num_classes")
    if configured_num_classes is not None and int(configured_num_classes) != len(class_names):
        raise ValueError(
            f"num_classes={configured_num_classes} does not match "
            f"{len(class_names)} configured classes."
        )

    dataset_root = Path(raw_data_dir) / _get(data_cfg, "dataset_dir", "digit_fabrics")
    dataset_seed = int(_get(data_cfg, "seed", 42) if seed is None else seed)

    split_samples = _make_splits(
        dataset_root=dataset_root,
        classes=class_names,
        samples_per_class=int(_get(data_cfg, "samples_per_class")),
        split_cfg=_get(data_cfg, "split"),
        seed=dataset_seed,
    )

    train_transform = build_transforms(data_cfg, train=True)
    eval_transform = build_transforms(data_cfg, train=False)
    train_flip_p = _horizontal_flip_p(data_cfg)

    return (
        DigitFabricsDataset(
            split_samples["train"],
            class_names,
            transform=train_transform,
            horizontal_flip_p=train_flip_p,
            seed=dataset_seed,
        ),
        DigitFabricsDataset(split_samples["val"], class_names, transform=eval_transform),
        DigitFabricsDataset(split_samples["test"], class_names, transform=eval_transform),
    )


def build_transforms(cfg: Any, train: bool) -> transforms.Compose:
    """Build image transforms for DIGIT fabric frames."""

    data_cfg = _data_cfg(cfg)
    input_cfg = _get(data_cfg, "input")
    normalization = _get(data_cfg, "normalization")
    augmentations = _get(data_cfg, "augmentations", {})
    use_augmentations = train and bool(_get(augmentations, "enabled", False))
    height = int(_get(input_cfg, "height"))
    width = int(_get(input_cfg, "width"))

    transform_list: list[Any] = []
    if use_augmentations and bool(_get(augmentations, "random_crop", False)):
        crop_scale = _float_range(
            augmentations,
            "random_crop_scale",
            default=(0.85, 1.0),
            min_value=0.0,
            max_value=1.0,
        )
        if crop_scale[0] <= 0.0:
            raise ValueError(f"random_crop_scale values must be > 0, got {crop_scale}.")
        crop_ratio = _float_range(
            augmentations,
            "random_crop_ratio",
            default=(width / height, width / height),
            min_value=0.0,
        )
        if crop_ratio[0] <= 0.0:
            raise ValueError(f"random_crop_ratio values must be > 0, got {crop_ratio}.")
        transform_list.append(
            transforms.RandomResizedCrop(
                size=(height, width),
                scale=crop_scale,
                ratio=crop_ratio,
            )
        )
    else:
        transform_list.append(transforms.Resize((height, width)))

    if use_augmentations and bool(_get(augmentations, "random_rotation", False)):
        transform_list.append(
            transforms.RandomRotation(
                degrees=_rotation_degrees(augmentations),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    if use_augmentations and bool(_get(augmentations, "color_jitter", False)):
        transform_list.append(
            transforms.ColorJitter(
                brightness=_non_negative_float(augmentations, "color_jitter_brightness", default=0.2),
                contrast=_non_negative_float(augmentations, "color_jitter_contrast", default=0.2),
                saturation=_non_negative_float(augmentations, "color_jitter_saturation", default=0.2),
                hue=_color_jitter_hue(augmentations),
            )
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(_get(normalization, "mean")),
                std=list(_get(normalization, "std")),
            ),
        ]
    )

    if use_augmentations and bool(_get(augmentations, "random_erasing", False)):
        erasing_scale = _float_range(
            augmentations,
            "random_erasing_scale",
            default=(0.02, 0.12),
            min_value=0.0,
            max_value=1.0,
        )
        if erasing_scale[0] <= 0.0:
            raise ValueError(f"random_erasing_scale values must be > 0, got {erasing_scale}.")
        erasing_ratio = _float_range(
            augmentations,
            "random_erasing_ratio",
            default=(0.3, 3.3),
            min_value=0.0,
        )
        if erasing_ratio[0] <= 0.0:
            raise ValueError(f"random_erasing_ratio values must be > 0, got {erasing_ratio}.")
        transform_list.append(
            transforms.RandomErasing(
                p=_probability(augmentations, "random_erasing_p", default=0.25),
                scale=erasing_scale,
                ratio=erasing_ratio,
                value=_get(augmentations, "random_erasing_value", 0.0),
            )
        )

    return transforms.Compose(transform_list)


def _make_splits(
    dataset_root: Path,
    classes: Sequence[str],
    samples_per_class: int,
    split_cfg: Any,
    seed: int,
) -> dict[str, list[ImageSample]]:
    _validate_split(split_cfg)
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}.")

    rng = random.Random(seed)
    splits: dict[str, list[ImageSample]] = {"train": [], "val": [], "test": []}

    for label, class_name in enumerate(classes):
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        image_paths = sorted(class_dir.glob("*.png"))
        if len(image_paths) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' has {len(image_paths)} PNG files, "
                f"but samples_per_class={samples_per_class}."
            )

        rng.shuffle(image_paths)
        selected_paths = image_paths[:samples_per_class]
        class_samples = [ImageSample(path=path, label=label) for path in selected_paths]

        train_count = int(samples_per_class * float(_get(split_cfg, "train")))
        val_count = int(samples_per_class * float(_get(split_cfg, "val")))
        test_start = train_count + val_count

        splits["train"].extend(class_samples[:train_count])
        splits["val"].extend(class_samples[train_count:test_start])
        splits["test"].extend(class_samples[test_start:])

    for samples in splits.values():
        rng.shuffle(samples)

    return splits


def _horizontal_flip_p(data_cfg: Any) -> float:
    augmentations = _get(data_cfg, "augmentations", {})
    if not bool(_get(augmentations, "enabled", False)):
        return 0.0
    if not bool(_get(augmentations, "horizontal_flip", False)):
        return 0.0

    probability = _probability(augmentations, "horizontal_flip_p", default=0.5)
    return probability


def _probability(cfg: Any, key: str, default: float) -> float:
    probability = float(_get(cfg, key, default))
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {probability}.")
    return probability


def _validate_split(split_cfg: Any) -> None:
    split_sum = sum(float(_get(split_cfg, key)) for key in ("train", "val", "test"))
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {split_sum}.")


def _data_cfg(cfg: Any) -> Any:
    if _has(cfg, "data"):
        return _get(cfg, "data")
    return cfg


def _has(cfg: Any, key: str) -> bool:
    if isinstance(cfg, Mapping):
        return key in cfg
    return hasattr(cfg, key)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _float_range(
    cfg: Any,
    key: str,
    default: tuple[float, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> tuple[float, float]:
    value = _get(cfg, key, default)
    try:
        values = tuple(float(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{key} must be a two-item numeric range, got {value!r}.") from exc

    if len(values) != 2:
        raise ValueError(f"{key} must be a two-item numeric range, got {value!r}.")
    lower, upper = values
    if lower > upper:
        raise ValueError(f"{key} lower bound must be <= upper bound, got {values}.")
    if min_value is not None and lower < min_value:
        raise ValueError(f"{key} values must be >= {min_value}, got {values}.")
    if max_value is not None and upper > max_value:
        raise ValueError(f"{key} values must be <= {max_value}, got {values}.")
    return values


def _rotation_degrees(cfg: Any) -> Union[float, tuple[float, float]]:
    value = _get(cfg, "random_rotation_degrees", 10.0)
    if isinstance(value, (int, float)):
        degrees = float(value)
        if degrees < 0.0:
            raise ValueError(f"random_rotation_degrees must be non-negative, got {degrees}.")
        return degrees

    degrees_range = _float_range(cfg, "random_rotation_degrees", default=(-10.0, 10.0))
    return degrees_range


def _non_negative_float(cfg: Any, key: str, default: float) -> float:
    value = float(_get(cfg, key, default))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative, got {value}.")
    return value


def _color_jitter_hue(cfg: Any) -> float:
    hue = float(_get(cfg, "color_jitter_hue", 0.02))
    if hue < 0.0 or hue > 0.5:
        raise ValueError(f"color_jitter_hue must be in [0, 0.5], got {hue}.")
    return hue
