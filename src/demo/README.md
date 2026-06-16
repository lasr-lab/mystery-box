# Demo Prototype

Temporary prototypes for live DIGIT tactile classification.

Use the project environment before running `python` commands:

```bash
source /home/max/miniforge3/bin/activate secai_demo_server
```

Run the OpenCV frontend from the repository root:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m src.demo.app demo=default model=mobilevit_s
```

Run the PySide6 desktop frontend from the repository root:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m src.demo.qt_app demo=default model=mobilevit_s
```

Install or update the project dependencies first if `PySide6` is not available in
the active environment.

OpenCV controls:

- `q` or `Esc`: quit
- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate

PySide6 controls:

- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate
- The `Reinitialize camera` button performs the same reset as `r`
- `q`, `Esc`, or `Ctrl+Q`: quit

Both frontends show the current frame prediction with its softmax scores, plus
the aggregate prediction over the last configured rolling window.

No-camera smoke check:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m compileall -q src/demo src/ML
```

## Docker Deployment

Build the CPU deployment image from the repository root:

```bash
docker build -t secai-demo-tactile .
```

By default, the container downloads the selected checkpoint from
`MaxHaufe/LASR-SECAI-DEMO` into `/app/models` at startup if it is missing.
To bake all three current checkpoints into the image during build instead:

```bash
docker build \
  --build-arg SECAI_DOWNLOAD_MODELS=true \
  -t secai-demo-tactile .
```

Run the PySide6 frontend with X11 forwarding and a DIGIT camera passed through:

```bash
xhost +local:docker
docker run --rm -it \
  --device=/dev/video2 \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  secai-demo-tactile \
  model=mobilevit_s \
  demo.sensor.device_path=/dev/video2
xhost -local:docker
```

Use `model=efficientnet_b0`, `model=mobilevit_s`, or `model=mobilevitv2_100`
to select one of the deployed checkpoints.
