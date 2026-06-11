"""Hydra entry point for DIGIT tactile classifier training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ML.data import create_datasets
from src.ML.models import create_model
from src.ML.trainer import Trainer

LOGGER = logging.getLogger(__name__)


def configure_logging(cfg: DictConfig) -> None:
    log_level = getattr(cfg.trainer, "log_level", "INFO")
    log_file = Path(str(getattr(cfg.trainer, "log_file", Path(cfg.paths.output_dir) / "train.log")))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    LOGGER.info("Logging to %s", log_file)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg)
    Trainer.set_seed(int(cfg.project.seed))
    LOGGER.info("Starting training: experiment=%s model=%s seed=%s", cfg.experiment_name, cfg.model.name, cfg.project.seed)
    train_dataset, val_dataset, test_dataset = create_datasets(cfg, cfg.paths.raw_data_dir, int(cfg.project.seed))
    LOGGER.info(
        "Datasets ready: train=%d val=%d test=%d classes=%s",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        train_dataset.classes,
    )
    model = create_model(cfg.model)
    trainer = Trainer(cfg, model, train_dataset, val_dataset, test_dataset)
    results = trainer.fit()
    LOGGER.info("Training finished:\n%s", OmegaConf.to_yaml(OmegaConf.create(results), resolve=True))


if __name__ == "__main__":
    main()
