"""NiceGUI frontend for the DIGIT tactile classification demo."""

from __future__ import annotations

import glob
import io
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from nicegui import app as nicegui_app
from nicegui import ui
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from starlette.responses import StreamingResponse

from src.demo.inference import DemoClassifier


@dataclass(frozen=True)
class FrameResult:
    """Background capture/inference result to apply on the UI event loop."""

    frame_jpeg: bytes
    status: str
    prediction: Optional[Any] = None
    probabilities: Optional[np.ndarray] = None
    clear_predictions: bool = False


class DigitTactileNiceGuiApp:
    """Timer-driven NiceGUI DIGIT tactile classification demo."""

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        self.demo_cfg = demo_cfg
        self.sensor_cfg = demo_cfg["sensor"]
        self.aggregate_window_frames = max(
            1, int(demo_cfg.get("aggregate_window_frames", 60))
        )

        self.classifier = self._load_classifier(model_cfg, data_cfg)
        self.class_names = [str(name) for name in self.classifier.class_names]
        if not self.class_names:
            raise RuntimeError("Demo classifier did not provide any class names.")

        self.camera: Optional[Any] = None
        self.camera_source: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.camera_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.video_condition = threading.Condition()
        self.tick_in_progress = False
        self.reinit_in_progress = False
        self.pending_frame_result: Optional[FrameResult] = None
        self.pending_reinit_status: Optional[str] = None
        self.latest_frame_jpeg = self._frame_to_jpeg(
            self._placeholder_frame("Starting camera")
        )
        self.latest_frame_version = 0
        self.shutdown_event = threading.Event()
        self.worker_threads: list[threading.Thread] = []
        self.probability_window: deque[np.ndarray] = deque(
            maxlen=self.aggregate_window_frames
        )
        self.status = "Starting demo."

        self.video: Any = None
        self.current_title: Any = None
        self.aggregate_title: Any = None
        self.status_label: Any = None
        self.camera_label: Any = None
        self.inference_label: Any = None
        self.current_rows: list[dict[str, Any]] = []
        self.aggregate_rows: list[dict[str, Any]] = []

    def build_ui(self) -> None:
        ui.colors(primary="#2563eb")
        ui.add_css(
            """
            body { background: #f6f7f9; }
            .demo-card { border: 1px solid #e5e7eb; box-shadow: none; }
            .class-label { width: 9.5rem; }
            .percent-label { width: 3.5rem; text-align: right; }
            """
        )

        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
            ui.label("DIGIT Tactile Demo").classes("text-2xl font-semibold")
            with ui.row().classes("w-full items-start gap-4"):
                with ui.card().classes("demo-card grow min-w-[320px]"):
                    self.video = ui.html(
                        '<img src="/digit_video_feed" '
                        'style="display:block;width:100%;border-radius:8px;'
                        'background:#000;" />',
                        sanitize=False,
                    )
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.button(
                            "Reinitialize camera",
                            on_click=self.reinitialize_camera,
                        ).props("unelevated")
                        ui.label("Press r to reinitialize").classes(
                            "text-sm text-gray-600"
                        )

                with ui.card().classes("demo-card w-full md:w-[430px]"):
                    ui.label("Runtime").classes("text-lg font-medium")
                    self.camera_label = ui.label("camera: -").classes("text-sm")
                    self.inference_label = ui.label(
                        f"inference: {self.classifier.device}"
                    ).classes("text-sm")
                    self.status_label = ui.label(self.status).classes(
                        "text-sm text-gray-700"
                    )

                    ui.separator()
                    self.current_title = ui.label("CURRENT: unavailable").classes(
                        "text-lg font-medium"
                    )
                    self.current_rows = self._build_probability_rows(ui, "#0ea5e9")

                    ui.separator()
                    self.aggregate_title = ui.label(
                        f"AGG 0/{self.aggregate_window_frames}: unavailable"
                    ).classes("text-lg font-medium")
                    self.aggregate_rows = self._build_probability_rows(ui, "#16a34a")

        ui.keyboard(on_key=self._handle_key, repeating=False)
        ui.timer(1.0 / self._ui_fps(), self.tick)
        self.reinitialize_camera()

    def run(self) -> None:
        host = "127.0.0.1"
        port = 8080

        @ui.page("/")
        def index() -> None:
            self.build_ui()

        @nicegui_app.get("/digit_video_feed")
        def video_feed() -> StreamingResponse:
            return StreamingResponse(
                self._video_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        nicegui_app.on_shutdown(self.shutdown)
        print(f"Open http://{host}:{port} in a browser.")
        ui.run(
            host=host,
            port=port,
            title="DIGIT Tactile Demo",
            reload=False,
            show=False,
            show_welcome_message=False,
        )

    def tick(self) -> None:
        if self._apply_pending_reinit():
            return
        if self._apply_pending_frame():
            return
        if self.tick_in_progress or self.reinit_in_progress:
            return

        self.tick_in_progress = True
        self._start_worker(
            target=self._capture_worker,
            name="digit-capture",
        )

    def _capture_worker(self) -> None:
        try:
            result = self._build_frame_result()
        except Exception as exc:
            result = FrameResult(
                frame_jpeg=self._frame_to_jpeg(self._error_frame()),
                status=f"Frame update failed: {exc}",
                clear_predictions=True,
            )
        with self.result_lock:
            self.pending_frame_result = result
            self.tick_in_progress = False

    def _apply_pending_frame(self) -> bool:
        with self.result_lock:
            result = self.pending_frame_result
            self.pending_frame_result = None
        if result is None:
            return False
        self.status = result.status
        if result.clear_predictions:
            self.probability_window.clear()
        elif result.probabilities is not None:
            self.probability_window.append(result.probabilities)

        self._set_latest_frame_jpeg(result.frame_jpeg)
        self._update_predictions(
            result.prediction,
            result.probabilities,
            self._aggregate_probabilities(),
        )
        self._update_runtime_labels()
        return True

    def reinitialize_camera(self) -> None:
        if self.reinit_in_progress:
            return

        self.reinit_in_progress = True
        self.status = "Reinitializing camera..."
        self._set_latest_frame_jpeg(
            self._frame_to_jpeg(self._placeholder_frame("Reinitializing camera"))
        )
        self.probability_window.clear()
        self._update_predictions(None, None, self._aggregate_probabilities())
        self._update_runtime_labels()

        self._start_worker(
            target=self._reinitialize_worker,
            name="digit-reinit",
        )

    def _reinitialize_worker(self) -> None:
        try:
            status = self._open_camera()
        except (AttributeError, OSError, RuntimeError, cv2.error) as exc:
            status = f"Camera reinit failed: {exc}"
            self.release_camera()
        with self.result_lock:
            self.pending_reinit_status = status
            self.reinit_in_progress = False

    def _apply_pending_reinit(self) -> bool:
        with self.result_lock:
            status = self.pending_reinit_status
            self.pending_reinit_status = None
        if status is None:
            return False
        self.status = status
        self.probability_window.clear()
        self._update_predictions(None, None, self._aggregate_probabilities())
        self._update_runtime_labels()
        return True

    def release_camera(self) -> None:
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    def shutdown(self) -> None:
        self.shutdown_event.set()
        with self.video_condition:
            self.video_condition.notify_all()
        for thread in tuple(self.worker_threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.release_camera()

    def _start_worker(self, target: Any, name: str) -> None:
        thread = threading.Thread(target=target, name=name)
        self.worker_threads = [
            existing for existing in self.worker_threads if existing.is_alive()
        ]
        self.worker_threads.append(thread)
        thread.start()

    def _load_classifier(
        self,
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> Any:
        return DemoClassifier(
            str(self.demo_cfg["model_checkpoint"]),
            model_cfg,
            data_cfg,
            device=str(self.demo_cfg.get("device", "auto")),
        )

    def _open_camera(self) -> str:
        with self.camera_lock:
            old_camera = self.camera
            self.camera = None
            if old_camera is not None:
                old_camera.release()

            source = self._camera_source()
            backend_name = str(self.sensor_cfg.get("backend", "CAP_V4L2"))
            backend = getattr(cv2, backend_name)
            capture_source = self._opencv_capture_source(source, backend)
            camera = cv2.VideoCapture(capture_source, backend)

            try:
                fourcc = cv2.VideoWriter_fourcc(*str(self.sensor_cfg["fourcc"])[:4])
                camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.sensor_cfg["width"]))
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.sensor_cfg["height"]))
                camera.set(cv2.CAP_PROP_FPS, int(self.sensor_cfg["fps"]))

                if not camera.isOpened():
                    raise RuntimeError(f"Could not open DIGIT camera source {source}.")

                self._warm_camera(camera, source)
            except Exception:
                camera.release()
                raise

            self.camera = camera
            self.camera_source = source
            status = (
                f"Camera {source}: "
                f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                f"{camera.get(cv2.CAP_PROP_FPS):.1f} fps, "
                f"{self._fourcc_to_string(int(camera.get(cv2.CAP_PROP_FOURCC)))}"
            )
            print(status)
            return status

    def _warm_camera(self, camera: Any, source: str) -> None:
        warmup_frames = int(self.sensor_cfg["warmup_frames"])
        if warmup_frames <= 0:
            return

        read_success = False
        for _ in range(warmup_frames):
            ok, _ = camera.read()
            read_success = read_success or ok

        if not read_success:
            raise RuntimeError(
                f"Could not read warmup frames from DIGIT camera source {source}."
            )

    def _camera_source(self) -> str:
        device_path = self.sensor_cfg.get("device_path")
        if device_path:
            return str(device_path)

        device_name = str(self.sensor_cfg["device_name"])
        source = self._find_video_device(device_name)
        if source is None:
            raise RuntimeError(
                f"No video device matching {device_name!r}. "
                "Set demo.sensor.device_path=/dev/videoX."
            )
        return source

    @staticmethod
    def _find_video_device(device_name: str) -> Optional[str]:
        matches: list[tuple[int, str]] = []
        for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            name_path = Path(name_file)
            try:
                actual_name = name_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if device_name.lower() not in actual_name.lower():
                continue
            matches.append(
                (
                    DigitTactileNiceGuiApp._device_index(name_path.parent),
                    f"/dev/{name_path.parent.name}",
                )
            )

        return sorted(matches)[0][1] if matches else None

    @staticmethod
    def _opencv_capture_source(source: str, backend: int) -> Any:
        if backend != cv2.CAP_V4L2:
            return source

        video_device = re.fullmatch(r"/dev/video(\d+)", source)
        if video_device is not None:
            return int(video_device.group(1))

        if source.isdecimal():
            return int(source)

        return source

    @staticmethod
    def _device_index(video_dir: Path) -> int:
        try:
            return int((video_dir / "index").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 999

    @staticmethod
    def _fourcc_to_string(value: int) -> str:
        if value <= 0:
            return "unknown"
        return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip()

    def _read_frame(self) -> Optional[np.ndarray]:
        with self.camera_lock:
            if self.camera is None:
                return None

            try:
                ok, frame = self.camera.read()
            except cv2.error:
                return None

            if not ok or frame is None:
                return None
            return frame

    def _build_frame_result(self) -> FrameResult:
        frame = self._read_frame()
        if frame is None:
            return FrameResult(
                frame_jpeg=self._frame_to_jpeg(self._error_frame()),
                status="Frame read failed. Press r or the button to reinit.",
                clear_predictions=True,
            )

        try:
            self._require_expected_size(frame)
            self.last_frame = frame.copy()
            prediction = self.classifier.predict(frame)
            probabilities = self._prediction_probabilities(prediction)
        except Exception as exc:
            return FrameResult(
                frame_jpeg=self._frame_to_jpeg(self._error_frame()),
                status=f"Inference failed: {exc}",
                clear_predictions=True,
            )

        if bool(self.demo_cfg.get("show_sensor_preview", True)):
            display_frame = frame
        else:
            display_frame = self._placeholder_frame("Sensor preview disabled")
        return FrameResult(
            frame_jpeg=self._frame_to_jpeg(display_frame),
            status=self.status,
            prediction=prediction,
            probabilities=probabilities,
        )

    def _prediction_probabilities(self, prediction: Any) -> np.ndarray:
        probabilities = prediction.probabilities
        if isinstance(probabilities, dict):
            probabilities = [probabilities[name] for name in self.class_names]
        if hasattr(probabilities, "detach"):
            probabilities = probabilities.detach().cpu().numpy()

        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if values.shape[0] != len(self.class_names):
            raise RuntimeError(
                "Prediction probability count does not match class count: "
                f"{values.shape[0]} != {len(self.class_names)}."
            )
        return values

    def _aggregate_probabilities(self) -> np.ndarray:
        if not self.probability_window:
            return np.zeros(len(self.class_names), dtype=np.float32)
        return np.mean(np.stack(tuple(self.probability_window), axis=0), axis=0)

    def _require_expected_size(self, frame: Any) -> None:
        expected_width = int(self.sensor_cfg["width"])
        expected_height = int(self.sensor_cfg["height"])
        height, width = frame.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"Expected {expected_width}x{expected_height} from DIGIT, "
                f"got {width}x{height}."
            )

    def _build_probability_rows(self, ui: Any, color: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for class_name in self.class_names:
            with ui.row().classes("w-full items-center gap-2"):
                label = ui.label(class_name).classes("class-label text-sm")
                bar = ui.linear_progress(value=0.0, show_value=False).classes("grow")
                bar.props("instant-feedback")
                bar.style(f"--q-primary: {color};")
                percent = ui.label("0%").classes("percent-label text-sm")
            rows.append({"label": label, "bar": bar, "percent": percent})
        return rows

    def _update_predictions(
        self,
        prediction: Any,
        probabilities: Optional[np.ndarray],
        aggregate_probabilities: np.ndarray,
    ) -> None:
        if self.current_title is None:
            return

        if probabilities is None:
            self.current_title.set_text("CURRENT: unavailable")
            self._update_probability_rows(
                self.current_rows,
                np.zeros(len(self.class_names)),
            )
        else:
            current_label = self._prediction_label(prediction, probabilities)
            current_confidence = float(np.max(probabilities))
            self.current_title.set_text(
                f"CURRENT: {current_label} ({current_confidence:.0%})"
            )
            self._update_probability_rows(self.current_rows, probabilities)

        if self.probability_window:
            aggregate_label = self._label_for_probabilities(aggregate_probabilities)
            aggregate_confidence = float(np.max(aggregate_probabilities))
            self.aggregate_title.set_text(
                f"AGG {len(self.probability_window)}/{self.aggregate_window_frames}: "
                f"{aggregate_label} ({aggregate_confidence:.0%})"
            )
        else:
            self.aggregate_title.set_text(
                f"AGG 0/{self.aggregate_window_frames}: unavailable"
            )
        self._update_probability_rows(self.aggregate_rows, aggregate_probabilities)

    def _update_probability_rows(
        self,
        rows: list[dict[str, Any]],
        probabilities: np.ndarray,
    ) -> None:
        winner_index = int(np.argmax(probabilities)) if len(probabilities) else 0
        for index, row in enumerate(rows):
            probability = float(np.clip(probabilities[index], 0.0, 1.0))
            row["bar"].set_value(probability)
            row["percent"].set_text(f"{probability:.0%}")
            row["label"].classes(
                replace=(
                    "class-label text-sm font-semibold"
                    if index == winner_index
                    else "class-label text-sm"
                )
            )

    def _update_runtime_labels(self) -> None:
        if self.status_label is None:
            return
        self.camera_label.set_text(f"camera: {self.camera_source or '-'}")
        self.inference_label.set_text(f"inference: {self.classifier.device}")
        self.status_label.set_text(self.status)
        self.status_label.classes(
            replace=(
                "text-sm text-orange-700"
                if self._status_is_error()
                else "text-sm text-gray-700"
            )
        )

    def _set_latest_frame_jpeg(self, frame_jpeg: bytes) -> None:
        with self.video_condition:
            self.latest_frame_jpeg = frame_jpeg
            self.latest_frame_version += 1
            self.video_condition.notify_all()

    def _video_stream(self) -> Any:
        last_version = -1
        while not self.shutdown_event.is_set():
            with self.video_condition:
                self.video_condition.wait_for(
                    lambda: (
                        self.latest_frame_version != last_version
                        or self.shutdown_event.is_set()
                    ),
                    timeout=1.0,
                )
                if self.shutdown_event.is_set():
                    break
                frame_jpeg = self.latest_frame_jpeg
                last_version = self.latest_frame_version

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame_jpeg)}\r\n\r\n".encode("ascii")
                + frame_jpeg
                + b"\r\n"
            )

    def _frame_to_jpeg(self, frame_bgr: np.ndarray) -> bytes:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        buffer = io.BytesIO()
        Image.fromarray(frame_rgb).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def _placeholder_frame(self, message: str) -> np.ndarray:
        width = int(self.sensor_cfg["width"])
        height = int(self.sensor_cfg["height"])
        frame = np.full((height, width, 3), (12, 12, 12), dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (20, max(30, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _error_frame(self) -> np.ndarray:
        frame = (
            self.last_frame.copy()
            if self.last_frame is not None
            else self._placeholder_frame("Frame unavailable")
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "Frame unavailable - press r or Reinitialize camera",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 190, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _handle_key(self, event: Any) -> None:
        if not getattr(getattr(event, "action", None), "keydown", False):
            return
        if getattr(getattr(event, "action", None), "repeat", False):
            return
        key = getattr(event, "key", None)
        name = str(getattr(key, "name", key or ""))
        if name.lower() == "r":
            if self.reinit_in_progress:
                return
            self.reinitialize_camera()

    def _ui_fps(self) -> float:
        sensor_fps = max(1, int(self.sensor_cfg.get("fps", 30)))
        return float(min(sensor_fps, 15))

    def _prediction_label(self, prediction: Any, probabilities: np.ndarray) -> str:
        label = getattr(prediction, "label", None)
        if label:
            return str(label)
        return self._label_for_probabilities(probabilities)

    def _label_for_probabilities(self, probabilities: np.ndarray) -> str:
        return self.class_names[int(np.argmax(probabilities))]

    def _status_is_error(self) -> bool:
        status = self.status.lower()
        return any(marker in status for marker in ("failed", "unavailable", "could not"))


def _config_section(cfg: DictConfig, name: str) -> dict[str, Any]:
    section = cfg.get(name)
    if section is None:
        raise RuntimeError(
            f"No Hydra {name!r} config selected. "
            f"Run with `{name}=default` if this repository default is still disabled."
        )
    resolved = OmegaConf.to_container(section, resolve=True)
    if not isinstance(resolved, dict):
        raise RuntimeError(f"Hydra config section {name!r} must resolve to a mapping.")
    return resolved


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    app = DigitTactileNiceGuiApp(
        demo_cfg=_config_section(cfg, "demo"),
        model_cfg=_config_section(cfg, "model"),
        data_cfg=_config_section(cfg, "data"),
    )
    app.run()


if __name__ == "__main__":
    main()
