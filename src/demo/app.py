from __future__ import annotations

import glob
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


class DigitTactileDemoApp:
    """OpenCV DIGIT tactile classification demo."""

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        self.demo_cfg = demo_cfg
        self.sensor_cfg = demo_cfg["sensor"]
        self.gui_cfg = demo_cfg["gui"]
        self.window_name = str(self.gui_cfg["window_name"])
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
        self.window_ready = False
        self.probability_window: deque[np.ndarray] = deque(
            maxlen=self.aggregate_window_frames
        )
        self.status = "Starting demo."

    def run(self) -> None:
        self._open_camera()
        self._print_instructions()

        try:
            while True:
                frame = self._read_frame()
                if frame is None:
                    preview = self._draw_error_preview()
                    self._ensure_window(preview)
                    cv2.imshow(self.window_name, preview)
                    if self._handle_key(cv2.waitKey(self._wait_delay()) & 0xFF):
                        break
                    continue

                self._require_expected_size(frame)
                self.last_frame = frame.copy()
                prediction = self.classifier.predict(frame)
                probabilities = self._prediction_probabilities(prediction)
                self.probability_window.append(probabilities)
                aggregate_probabilities = self._aggregate_probabilities()

                preview = self._draw_preview(
                    frame,
                    prediction,
                    probabilities,
                    aggregate_probabilities,
                )
                self._ensure_window(preview)
                cv2.imshow(self.window_name, preview)

                if self._handle_key(cv2.waitKey(self._wait_delay()) & 0xFF):
                    break
        finally:
            self._release_camera()
            cv2.destroyAllWindows()

    def _load_classifier(
        self,
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> Any:
        try:
            from src.demo.inference import DemoClassifier
        except ImportError as exc:
            raise RuntimeError(
                "src.demo.inference.DemoClassifier is required for the OpenCV demo."
            ) from exc

        return DemoClassifier(
            str(self.demo_cfg["model_checkpoint"]),
            model_cfg,
            data_cfg,
            device=str(self.demo_cfg.get("device", "auto")),
        )

    def _open_camera(self) -> None:
        source = self._camera_source()
        backend_name = str(self.sensor_cfg.get("backend", "CAP_V4L2"))
        backend = getattr(cv2, backend_name)
        camera = cv2.VideoCapture(source, backend)

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

        old_camera = self.camera
        self.camera = camera
        self.camera_source = source
        self.window_ready = False
        self.probability_window.clear()
        self.status = (
            f"Camera {source}: "
            f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
            f"{camera.get(cv2.CAP_PROP_FPS):.1f} fps, "
            f"{self._fourcc_to_string(int(camera.get(cv2.CAP_PROP_FOURCC)))}"
        )
        if old_camera is not None:
            old_camera.release()
        print(self.status)

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
                    DigitTactileDemoApp._device_index(name_path.parent),
                    f"/dev/{name_path.parent.name}",
                )
            )

        return sorted(matches)[0][1] if matches else None

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

    def _release_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.window_ready = False

    def _handle_key(self, key: int) -> bool:
        if key == 255:
            return False
        if key in (27, ord("q"), ord("Q")):
            return True
        if key in (ord("r"), ord("R")):
            self._reinitialize_camera()
        return False

    def _reinitialize_camera(self) -> None:
        try:
            self._open_camera()
        except (AttributeError, OSError, RuntimeError, cv2.error) as exc:
            self.status = f"Camera reinit failed: {exc}"
            print(self.status)

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            self.status = "Camera unavailable. Press r to reinit."
            return None

        try:
            ok, frame = self.camera.read()
        except cv2.error as exc:
            self.status = f"Frame read failed: {exc}. Press r to reinit."
            return None

        if not ok or frame is None:
            self.status = "Frame read failed. Press r to reinit."
            return None
        return frame

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

    def _ensure_window(self, preview: np.ndarray) -> None:
        if self.window_ready:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        height, width = preview.shape[:2]
        scale = float(self.gui_cfg["display_scale"])
        cv2.resizeWindow(self.window_name, int(width * scale), int(height * scale))
        self.window_ready = True

    def _draw_preview(
        self,
        frame: np.ndarray,
        prediction: Any,
        probabilities: np.ndarray,
        aggregate_probabilities: np.ndarray,
    ) -> np.ndarray:
        panel_width = int(self.gui_cfg["side_panel_width"])
        panel = np.full(
            (frame.shape[0], panel_width, 3),
            (28, 28, 28),
            dtype=np.uint8,
        )
        self._draw_panel(panel, prediction, probabilities, aggregate_probabilities)

        if bool(self.demo_cfg.get("show_sensor_preview", True)):
            video = frame
        else:
            video = np.full_like(frame, (10, 10, 10))
            self._put_text(video, "Sensor preview disabled", 20, 30, (210, 210, 210))
        return np.hstack((panel, video))

    def _draw_error_preview(self) -> np.ndarray:
        frame = self._error_frame()
        panel_width = int(self.gui_cfg["side_panel_width"])
        panel = np.full(
            (frame.shape[0], panel_width, 3),
            (28, 28, 28),
            dtype=np.uint8,
        )
        self._draw_error_panel(panel)

        video = frame.copy()
        self._put_text(video, "Frame unavailable", 20, 30, (80, 190, 255), 0.55, 2)
        self._put_text(video, self._short_status(video.shape[1]), 20, 54)
        return np.hstack((panel, video))

    def _error_frame(self) -> np.ndarray:
        if self.last_frame is not None:
            return self.last_frame

        expected_width = int(self.sensor_cfg["width"])
        expected_height = int(self.sensor_cfg["height"])
        return np.full((expected_height, expected_width, 3), (10, 10, 10), dtype=np.uint8)

    def _draw_error_panel(self, panel: np.ndarray) -> None:
        height, width = panel.shape[:2]
        x = 12
        y = 16

        self._put_text(panel, "DIGIT TACTILE DEMO", x, y, (170, 220, 255), 0.42, 1)
        y += 15
        self._put_text(panel, f"camera: {self.camera_source or '-'}", x, y)
        y += 12
        self._put_text(panel, f"inference: {self.classifier.device}", x, y)
        y += 22
        self._put_text(panel, "CURRENT: unavailable", x, y, (245, 245, 245), 0.36, 1)
        y += 18
        self._put_text(panel, "No frame was read from the sensor.", x, y, (80, 190, 255))
        y += 14
        self._put_text(panel, "Press r to reinitialize.", x, y)

        self._put_text(panel, "r: reinit camera   q/esc: quit", x, height - 18)
        self._put_text(panel, self._short_status(width), x, height - 6, self._status_color())

    def _draw_panel(
        self,
        panel: np.ndarray,
        prediction: Any,
        probabilities: np.ndarray,
        aggregate_probabilities: np.ndarray,
    ) -> None:
        height, width = panel.shape[:2]
        x = 12
        y = 16
        bottom_reserved = 32

        self._put_text(panel, "DIGIT TACTILE DEMO", x, y, (170, 220, 255), 0.42, 1)
        y += 15
        self._put_text(panel, f"camera: {self.camera_source or '-'}", x, y)
        y += 12
        self._put_text(panel, f"inference: {self.classifier.device}", x, y)
        y += 15

        remaining_height = max(80, height - y - bottom_reserved)
        chart_height = max(28, (remaining_height - 28) // 2)

        current_label = self._prediction_label(prediction, probabilities)
        current_confidence = float(np.max(probabilities))
        self._put_text(
            panel,
            f"CURRENT: {current_label} ({current_confidence:.0%})",
            x,
            y,
            (245, 245, 245),
            0.36,
            1,
        )
        y += 10
        y = self._draw_probability_bars(
            panel,
            probabilities,
            x,
            y,
            width - (2 * x),
            chart_height,
            (70, 200, 255),
        )
        y += 8

        aggregate_label = self._label_for_probabilities(aggregate_probabilities)
        aggregate_confidence = float(np.max(aggregate_probabilities))
        self._put_text(
            panel,
            (
                f"AGG {len(self.probability_window)}/"
                f"{self.aggregate_window_frames}: "
                f"{aggregate_label} ({aggregate_confidence:.0%})"
            ),
            x,
            y,
            (245, 245, 245),
            0.36,
            1,
        )
        y += 10
        self._draw_probability_bars(
            panel,
            aggregate_probabilities,
            x,
            y,
            width - (2 * x),
            chart_height,
            (120, 220, 120),
        )

        self._put_text(panel, "r: reinit camera   q/esc: quit", x, height - 18)
        self._put_text(panel, self._short_status(width), x, height - 6, self._status_color())

    def _draw_probability_bars(
        self,
        panel: np.ndarray,
        probabilities: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        accent_color: tuple[int, int, int],
    ) -> int:
        row_count = len(self.class_names)
        row_height = max(7, height // row_count)
        bar_height = max(3, row_height - 4)
        label_width = min(118, width // 3)
        percent_width = 36
        bar_x = x + label_width
        bar_width = max(24, width - label_width - percent_width - 6)
        winner_index = int(np.argmax(probabilities))

        for index, class_name in enumerate(self.class_names):
            probability = float(np.clip(probabilities[index], 0.0, 1.0))
            row_y = y + (index * row_height)
            text_y = row_y + row_height - 2
            color = accent_color if index == winner_index else (110, 110, 110)

            self._put_text(
                panel,
                self._truncate(class_name, 16),
                x,
                text_y,
                (225, 225, 225),
                0.27,
            )
            cv2.rectangle(
                panel,
                (bar_x, row_y + 2),
                (bar_x + bar_width, row_y + 2 + bar_height),
                (55, 55, 55),
                -1,
            )
            cv2.rectangle(
                panel,
                (bar_x, row_y + 2),
                (bar_x + int(bar_width * probability), row_y + 2 + bar_height),
                color,
                -1,
            )
            self._put_text(
                panel,
                f"{probability:.0%}",
                bar_x + bar_width + 5,
                text_y,
                (210, 210, 210),
                0.27,
            )

        return y + (row_count * row_height)

    def _prediction_label(self, prediction: Any, probabilities: np.ndarray) -> str:
        label = getattr(prediction, "label", None)
        if label:
            return str(label)
        return self._label_for_probabilities(probabilities)

    def _label_for_probabilities(self, probabilities: np.ndarray) -> str:
        return self.class_names[int(np.argmax(probabilities))]

    def _status_color(self) -> tuple[int, int, int]:
        status = self.status.lower()
        if "failed" in status or "unavailable" in status or "could not" in status:
            return (80, 190, 255)
        if self.status.startswith("Camera"):
            return (120, 220, 120)
        if self.status.startswith("Frame"):
            return (80, 190, 255)
        return (210, 210, 210)

    def _short_status(self, panel_width: int) -> str:
        max_chars = max(12, panel_width // 8)
        return self._truncate(self.status, max_chars)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = (210, 210, 210),
        scale: float = 0.32,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _wait_delay(self) -> int:
        return int(self.gui_cfg["wait_key_delay_ms"])

    def _print_instructions(self) -> None:
        print(f"Checkpoint: {self.demo_cfg['model_checkpoint']}")
        print(f"Inference device: {self.classifier.device}")
        print(f"Aggregate window: {self.aggregate_window_frames} frame(s)")
        print(f"Classes: {', '.join(self.class_names)}")
        print("Press r to reinitialize the sensor. Press q or esc to quit.")


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
    app = DigitTactileDemoApp(
        demo_cfg=_config_section(cfg, "demo"),
        model_cfg=_config_section(cfg, "model"),
        data_cfg=_config_section(cfg, "data"),
    )
    app.run()


if __name__ == "__main__":
    main()
