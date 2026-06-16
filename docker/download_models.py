"""Download Hugging Face checkpoints into the demo model directory."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


CHECKPOINT_SUFFIXES = {".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
DEFAULT_REPO_ID = "MaxHaufe/LASR-SECAI-DEMO"
DEFAULT_MODELS_DIR = "/app/models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("SECAI_MODEL_REPO", DEFAULT_REPO_ID))
    parser.add_argument("--revision", default=os.environ.get("SECAI_MODEL_REVISION", "main"))
    parser.add_argument("--models-dir", default=os.environ.get("SECAI_MODELS_DIR", DEFAULT_MODELS_DIR))
    parser.add_argument(
        "--required-model",
        action="append",
        default=[],
        help="Hydra model name or checkpoint filename expected under models-dir.",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Do not contact Hugging Face if all required checkpoints already exist.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    required_checkpoints = [_checkpoint_filename(model) for model in args.required_model]

    if args.skip_if_present and _has_required_checkpoints(models_dir, required_checkpoints):
        return

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(models_dir),
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=[
            "*.ckpt",
            "*.json",
            "*.onnx",
            "*.pt",
            "*.pth",
            "*.safetensors",
            "*.yaml",
            "*.yml",
            "README.md",
        ],
    )

    _promote_nested_checkpoints(models_dir, required_checkpoints)

    missing = [name for name in required_checkpoints if not (models_dir / name).is_file()]
    if missing:
        available = "\n".join(
            f"  - {path.relative_to(models_dir)}" for path in _checkpoint_files(models_dir)
        )
        raise SystemExit(
            "Missing expected checkpoint(s) after Hugging Face download: "
            f"{', '.join(missing)}\nAvailable checkpoint files:\n{available or '  - none'}"
        )


def _checkpoint_filename(model_or_filename: str) -> str:
    value = model_or_filename.strip()
    if not value:
        raise ValueError("--required-model cannot be empty.")
    if Path(value).suffix in CHECKPOINT_SUFFIXES:
        return Path(value).name
    return f"{value}.pt"


def _has_required_checkpoints(models_dir: Path, required_checkpoints: list[str]) -> bool:
    return bool(required_checkpoints) and all((models_dir / name).is_file() for name in required_checkpoints)


def _promote_nested_checkpoints(models_dir: Path, required_checkpoints: list[str]) -> None:
    if not required_checkpoints:
        return

    by_name = {path.name: path for path in _checkpoint_files(models_dir)}
    for filename in required_checkpoints:
        destination = models_dir / filename
        if destination.is_file():
            continue
        source = by_name.get(filename)
        if source is None:
            continue
        shutil.copy2(source, destination)


def _checkpoint_files(models_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in models_dir.rglob("*")
        if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES
    )


if __name__ == "__main__":
    main()
