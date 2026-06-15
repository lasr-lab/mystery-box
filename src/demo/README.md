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

Run the NiceGUI frontend from the repository root:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m src.demo.nicegui_app demo=default model=mobilevit_s
```

Then open `http://127.0.0.1:8080` in a browser.

Run the PySide6 desktop frontend from the repository root:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m src.demo.qt_app demo=default model=mobilevit_s
```

Install or update the project dependencies first if `PySide6` is not available in
the active environment.

OpenCV controls:

- `q` or `Esc`: quit
- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate

NiceGUI controls:

- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate
- The `Reinitialize camera` button performs the same reset as `r`

PySide6 controls:

- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate
- The `Reinitialize camera` button performs the same reset as `r`
- `q`, `Esc`, or `Ctrl+Q`: quit

All frontends show the current frame prediction with its softmax scores, plus
the aggregate prediction over the last configured rolling window.

No-camera smoke check:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m compileall -q src/demo src/ML
```
