# AGENTS.md

## Project Overview

This repository is for a DIGIT tactile sensor demo. The goal is to classify tactile interactions with a 3D-printed sensor against fabrics mounted on a 3D-printed board.

The classification problem has 6 classes:
- `fabric_01`
- `fabric_02`
- `fabric_03`
- `fabric_04`
- `fabric_05`
- `nothing`

Rename the fabric class labels once the real fabric names are known.

## Expected Repository Structure

```text
.
├── AGENTS.md
├── config/
│   ├── config.yaml
│   ├── data/
│   ├── demo/
│   ├── model/
│   └── trainer/
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
└── src/
    ├── demo/
    └── ML/
```

## Directory Responsibilities

- `src/demo/`: Demo-facing code, including the web interface for live or recorded DIGIT tactile classification.
- `src/ML/`: Machine learning code, including training, evaluation, model definitions, dataset handling, and inference utilities.
- `config/`: Hydra configuration files. Keep runtime parameters in YAML, not hard-coded in Python.
- `data/raw/`: Raw captures from the DIGIT tactile sensor. Do not mutate these files in processing scripts.
- `data/processed/`: Derived datasets ready for training or evaluation.
- `data/external/`: Third-party or manually supplied assets that are not produced by this codebase.

## Configuration Policy

Use Hydra for experiment, training, and demo configuration.

- Root config: `config/config.yaml`
- Data config group: `config/data/`
- Model config group: `config/model/`
- Trainer config group: `config/trainer/`
- Demo config group: `config/demo/`

Prefer adding a new YAML file to a config group over adding command-line flags or constants in source code.

## Development Notes

- Keep data collection, preprocessing, training, inference, and demo UI concerns separated.
- Treat the `nothing` class as a first-class label, not as missing data.
- Keep raw sensor captures reproducible by recording sensor settings, class label, board position if relevant, and capture timestamp.
- Avoid committing large data files or model checkpoints unless explicitly requested.
- Use clear names for runs and checkpoints so demo models can be traced back to training configs.

## Python Conventions

- This project currently keeps the package name `src/ML/` because that was requested. If refactoring later, prefer lowercase package names such as `src/ml/`.
- Keep trainer orchestration in `src/ML/trainer.py` and entry points thin.
- Keep demo code in `src/demo/` and avoid importing training-only dependencies in the demo path unless required.
