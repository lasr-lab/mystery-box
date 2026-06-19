# Mystery Box Tactile Demo

DIGIT tactile sensor demo for classifying fabrics.

The 3D-printing hardware counterpart for this demo is available at https://github.com/lasr-lab/mystery-box-hardware.

## Deployment and Running the Demo With Docker

Docker is used for deploying and running the demo.

```bash
docker build \
  --build-arg SECAI_DOWNLOAD_MODELS=true \
  -t secai-demo-tactile \
  https://github.com/lasr-lab/mystery-box.git
```

Run the Qt demo with access to the DIGIT camera and local X11 display:

```bash
xhost +local:docker

docker run --rm -it \
  --device=/dev/video2 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  secai-demo-tactile \
  model=mobilevit_s \
  demo.sensor.device_path=/dev/video2

xhost -local:docker
```

Available model configs are `efficientnet_b0`, `mobilevit_s`, and `mobilevitv2_100`.
You might need to reconfigure the device path to point to another `/dev/videoX`.

If for some reason, the models should not be pre-downloaded, use this instead:

```bash
docker build -t secai-demo-tactile https://github.com/lasr-lab/mystery-box.git
```

This will download the models at container startup instead of build time.

## Repository Structure

```text
config/                  Hydra configs for data, models, trainer, demo, collection
src/datacollection/      DIGIT image capture UI
src/demo/                Qt and OpenCV live demo frontends
src/ML/                  datasets, model factory, training, inference utilities
data/raw/digit_fabrics/  raw class folders used for training
models/                  pretrained demo checkpoints, download from HF
outputs/                 Hydra training runs and trained checkpoints
```

Data should end up in `data/raw/digit_fabrics/<class_name>/*.png`.

Demo model files should end up in `models/efficientnet_b0.pt`, `models/mobilevit_s.pt`, and `models/mobilevitv2_100.pt`.

## Installation of Environment

Create and activate the local development environment:

```bash
mamba env create -f environment.yaml
mamba activate secai_demo_server
```

## Get Data

Download the dataset from Google Drive:

https://drive.google.com/drive/folders/1sOHJcTk-RO4Zir7QYIEkrYjEx-Db_idm?usp=sharing

Extract it from the repository root so the final path is `data/raw/digit_fabrics/`.

Expected result:

```text
data/raw/digit_fabrics/nothing/*.png
data/raw/digit_fabrics/cotton/*.png
data/raw/digit_fabrics/wool/*.png
...
```

## Get Pretrained Models

Download pretrained checkpoints from Hugging Face:

https://huggingface.co/MaxHaufe/LASR-SECAI-DEMO/tree/main

Place the `.pt` files in `models/`:

```bash
python -m pip install huggingface_hub
mkdir -p models
huggingface-cli download MaxHaufe/LASR-SECAI-DEMO \
  --local-dir models \
  --local-dir-use-symlinks False
```

Expected result:

```text
models/efficientnet_b0.pt
models/mobilevit_s.pt
models/mobilevitv2_100.pt
```

## Run Demo

Local Qt demo:

```bash
mamba activate secai_demo_server
python -m src.demo.qt_app demo=default model=mobilevit_s
```

Local OpenCV demo:

```bash
mamba activate secai_demo_server
python -m src.demo.app demo=default model=mobilevit_s
```

If the DIGIT camera is not auto-detected, pass the device explicitly:

```bash
python -m src.demo.qt_app demo=default model=mobilevit_s demo.sensor.device_path=/dev/video2
```

Alternatively, change the config

Available model configs are `efficientnet_b0`, `mobilevit_s`, and `mobilevitv2_100`.

## Run Data Collection

Collect DIGIT frames into `data/raw/digit_fabrics/<class_name>/`:

```bash
mamba activate secai_demo_server
python -m src.datacollection.collector datacollection=default
```

Keys: `0` nothing, `1` cotton, `2` wool, `3` curdory, `4` synthetic leather, `5` teddy, `6` flower fabric, `7` 3D print, `8` finger. Press `r` to reinitialize the camera and `q` or `Esc` to quit.

## Train Models

Train from `data/raw/digit_fabrics/`:

```bash
mamba activate secai_demo_server
python -m src.ML.train model=mobilevit_s
```

Checkpoints are written under `outputs/<run>/checkpoints/<model>.pt`. Copy a trained checkpoint into `models/<model>.pt` to use it in the demo.
