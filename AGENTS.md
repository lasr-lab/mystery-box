# AGENTS.md

## Project Overview

This repository is for a DIGIT tactile sensor demo. The goal is to classify tactile interactions with a 3D-printed sensor against fabrics mounted on a 3D-printed board.

The classification problem currently has 9 classes:
- `nothing` (displayed as no contact during collection)
- `cotton`
- `wool`
- `curdory`
- `synthetic_leather`
- `teddy`
- `flower_fabric`
- `3dprint`
- `finger`

The data collection key mapping is:
- `0`: `nothing` / no contact
- `1`: `cotton`
- `2`: `wool`
- `3`: `curdory`
- `4`: `synthetic_leather`
- `5`: `teddy`
- `6`: `flower_fabric`
- `7`: `3dprint`
- `8`: `finger`

## Default Environment

- Default Python interpreter: `/home/max/miniforge3/envs/secai_demo_server/bin/python`
- Activate before running project commands: `source /home/max/miniforge3/bin/activate secai_demo_server`

## Expected Repository Structure

```text
.
├── AGENTS.md
├── config/
│   ├── config.yaml
│   ├── data/
│   ├── datacollection/
│   ├── demo/
│   ├── model/
│   └── trainer/
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
└── src/
    ├── datacollection/
    ├── demo/
    └── ML/
```

## Directory Responsibilities

- `src/datacollection/`: Data collection code for raw DIGIT captures, including GUI-based single-frame capture tools.
- `src/demo/`: Demo-facing code, including the desktop interfaces for live or recorded DIGIT tactile classification.
- `src/ML/`: Machine learning code, including training, evaluation, model definitions, dataset handling, and inference utilities.
- `config/`: Hydra configuration files. Keep runtime parameters in YAML, not hard-coded in Python.
- `data/raw/`: Raw captures from the DIGIT tactile sensor. Do not mutate these files in processing scripts.
- `data/processed/`: Derived datasets ready for training or evaluation.
- `data/external/`: Third-party or manually supplied assets that are not produced by this codebase.

## Configuration Policy

Use Hydra for experiment, training, and demo configuration.

- Root config: `config/config.yaml`
- Data config group: `config/data/`
- Data collection config group: `config/datacollection/`
- Model config group: `config/model/`
- Trainer config group: `config/trainer/`
- Demo config group: `config/demo/`

Prefer adding a new YAML file to a config group over adding command-line flags or constants in source code.

## Development Notes

- Keep data collection, preprocessing, training, inference, and demo UI concerns separated.
- Treat the `nothing` class as a first-class label, not as missing data.
- Store data collection captures in class-specific subdirectories under `data/raw/` and keep collection parameters in `config/datacollection/`.
- Keep raw sensor captures reproducible by recording sensor settings, class label, board position if relevant, and capture timestamp.
- Avoid committing large data files or model checkpoints unless explicitly requested.
- Use clear names for runs and checkpoints so demo models can be traced back to training configs.

## Demo UI Style

- Follow the SECAI visual identity from `secai.org`: light/white surfaces, blue typography, cyan/teal primary accents, green secondary accents, clean rounded cards, and subtle dotted/composite motifs where useful.
- Avoid unrelated visual directions such as purple defaults, generic yellow warning boxes, or dark toast overlays unless explicitly requested.
- Warning and guidance overlays should feel like SECAI-branded status cards, not operating-system alerts.

## Python Conventions

- This project currently keeps the package name `src/ML/` because that was requested. If refactoring later, prefer lowercase package names such as `src/ml/`.
- Keep trainer orchestration in `src/ML/trainer.py` and entry points thin.
- Keep demo code in `src/demo/` and avoid importing training-only dependencies in the demo path unless required.
- Keep data collection entry points in `src/datacollection/` and avoid importing training-only dependencies there.
- Prefer fail-fast behavior across the project: do not add defensive import fallbacks or broad exception handling that hides dependency, config, camera, model, data, or inference errors.
- Avoid one-off helper functions; inline logic at the call site unless the helper is reused or isolates a substantial, cohesive operation.
