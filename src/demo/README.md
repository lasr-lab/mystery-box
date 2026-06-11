# Demo Prototype

Temporary OpenCV prototype for live DIGIT tactile classification. It uses
`models/mobilevit_s.pt` by default.

Run from the repository root with the project Python/Hydra entry point:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m src.demo.app demo=default model=mobilevit_s
```

Controls:

- `q` or `Esc`: quit
- `r`: reinitialize the camera/sensor; this also resets the rolling prediction aggregate

The left panel shows the current frame prediction with its softmax scores, plus
the aggregate prediction over the last 60 frames.

No-camera smoke check:

```bash
/home/max/miniforge3/envs/secai_demo_server/bin/python -m compileall -q src/demo src/ML
```
