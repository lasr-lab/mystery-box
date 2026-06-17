from __future__ import annotations

import glob
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.demo.inference import DemoClassifier


class DigitTactileDemoApp:
    """OpenCV DIGIT tactile classification demo."""

    MIN_SIDE_PANEL_WIDTH = 360

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        self.demo_cfg = demo_cfg
        self.sensor_cfg = demo_cfg["sensor"]
        self.gui_cfg = demo_cfg["gui"]
        self._validate_gui_config()
        self.window_name = str(self.gui_cfg["window_name"])
        self.aggregate_window_frames = max(
            1, int(demo_cfg.get("aggregate_window_frames", 60))
        )

        self.classifier = DemoClassifier(
            str(self.demo_cfg["model_checkpoint"]),
            model_cfg,
            data_cfg,
            device=str(self.demo_cfg.get("device", "auto")),
        )
        self.class_names = [str(name) for name in self.classifier.class_names]
        if not self.class_names:
            raise RuntimeError("Demo classifier did not provide any class names.")

        self.camera: Optional[Any] = None
        self.camera_source: Optional[str] = None
        self.window_ready = False
        self.probability_window: deque[np.ndarray] = deque(
            maxlen=self.aggregate_window_frames
        )
        self.status = "Starting demo."

    def _validate_gui_config(self) -> None:
        try:
            side_panel_width = int(self.gui_cfg["side_panel_width"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "demo.gui.side_panel_width must be an integer "
                f">= {self.MIN_SIDE_PANEL_WIDTH}."
            ) from exc
        if side_panel_width < self.MIN_SIDE_PANEL_WIDTH:
            raise RuntimeError(
                "demo.gui.side_panel_width is too small for the OpenCV demo "
                f"layout: got {side_panel_width}, minimum is "
                f"{self.MIN_SIDE_PANEL_WIDTH}."
            )

        try:
            display_scale = float(self.gui_cfg["display_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "demo.gui.display_scale must be a finite positive number."
            ) from exc
        if not np.isfinite(display_scale) or display_scale <= 0.0:
            raise RuntimeError(
                "demo.gui.display_scale must be a finite positive number."
            )

        try:
            wait_key_delay_ms = int(self.gui_cfg["wait_key_delay_ms"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "demo.gui.wait_key_delay_ms must be a positive integer. "
                "Use at least 1 ms so the live preview does not block."
            ) from exc
        if wait_key_delay_ms < 1:
            raise RuntimeError(
                "demo.gui.wait_key_delay_ms must be a positive integer. "
                "Use at least 1 ms so the live preview does not block."
            )

    def run(self) -> None:
        self._open_camera()
        self._print_instructions()

        try:
            while True:
                frame = self._read_frame()
                self._require_expected_size(frame)
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

    def _open_camera(self) -> None:
        old_camera = self.camera
        self.camera = None
        if old_camera is not None:
            old_camera.release()

        source = self._camera_source()
        backend_name = str(self.sensor_cfg.get("backend", "CAP_V4L2"))
        backend = getattr(cv2, backend_name)
        capture_source = self._opencv_capture_source(source, backend)
        camera = cv2.VideoCapture(capture_source, backend)
        camera_ready = False

        try:
            if not camera.isOpened():
                raise RuntimeError(f"Could not open DIGIT camera source {source}.")

            fourcc = cv2.VideoWriter_fourcc(*str(self.sensor_cfg["fourcc"])[:4])
            if not camera.set(cv2.CAP_PROP_FOURCC, fourcc):
                raise RuntimeError(f"Could not set demo.sensor.fourcc for {source}.")
            if not camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.sensor_cfg["width"])):
                raise RuntimeError(f"Could not set demo.sensor.width for {source}.")
            if not camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.sensor_cfg["height"])):
                raise RuntimeError(f"Could not set demo.sensor.height for {source}.")
            if not camera.set(cv2.CAP_PROP_FPS, int(self.sensor_cfg["fps"])):
                raise RuntimeError(f"Could not set demo.sensor.fps for {source}.")

            self._warm_camera(camera, source)
            status = (
                f"Camera {source}: "
                f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                f"{camera.get(cv2.CAP_PROP_FPS):.1f} fps, "
                f"{self._fourcc_to_string(int(camera.get(cv2.CAP_PROP_FOURCC)))}"
            )
            camera_ready = True
        finally:
            if not camera_ready:
                camera.release()

        self.camera = camera
        self.camera_source = source
        self.window_ready = False
        self.probability_window.clear()
        self.status = status
        print(self.status)

    def _warm_camera(self, camera: Any, source: str) -> None:
        warmup_frames = int(self.sensor_cfg["warmup_frames"])
        if warmup_frames < 0:
            raise RuntimeError("demo.sensor.warmup_frames must be non-negative.")
        if warmup_frames <= 0:
            return

        read_success = False
        for _ in range(warmup_frames):
            ok, frame = camera.read()
            read_success = read_success or (ok and frame is not None)

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
        expected_name = device_name.lower()
        for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            name_path = Path(name_file)
            actual_name = name_path.read_text(encoding="utf-8").strip()
            if expected_name not in actual_name.lower():
                continue
            matches.append(
                (
                    int((name_path.parent / "index").read_text(encoding="utf-8").strip()),
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
            self._open_camera()
        return False

    def _read_frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("DIGIT camera is not opened.")

        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Failed to read frame from DIGIT camera source {self.camera_source}."
            )
        return frame

    def _prediction_probabilities(self, prediction: Any) -> np.ndarray:
        probabilities = prediction.probabilities
        if isinstance(probabilities, dict):
            try:
                probabilities = [probabilities[name] for name in self.class_names]
            except KeyError as exc:
                raise RuntimeError(
                    f"Prediction probabilities are missing class {exc.args[0]!r}."
                ) from exc
        if hasattr(probabilities, "detach"):
            probabilities = probabilities.detach().cpu().numpy()

        try:
            values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "Prediction probabilities must be a numeric vector."
            ) from exc
        self._validate_probability_vector(values)
        return values

    def _validate_probability_vector(self, values: np.ndarray) -> None:
        if values.shape[0] != len(self.class_names):
            raise RuntimeError(
                "Prediction probability vector length does not match class count: "
                f"got {values.shape[0]}, expected {len(self.class_names)}."
            )

        invalid_indices = np.flatnonzero(~np.isfinite(values))
        if invalid_indices.size:
            invalid_values = values[invalid_indices].tolist()
            raise RuntimeError(
                "Prediction probability vector contains non-finite value(s) at "
                f"index(es) {invalid_indices.tolist()}: {invalid_values}."
            )

        invalid_indices = np.flatnonzero((values < 0.0) | (values > 1.0))
        if invalid_indices.size:
            invalid_values = values[invalid_indices].tolist()
            raise RuntimeError(
                "Prediction probability vector values must be in [0.0, 1.0]; "
                f"invalid index(es) {invalid_indices.tolist()}: {invalid_values}."
            )

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
        panel = self._new_panel(frame.shape[0], panel_width)
        self._draw_panel(panel, prediction, probabilities, aggregate_probabilities)

        if bool(self.demo_cfg.get("show_sensor_preview", True)):
            video = frame
        else:
            video = np.full_like(frame, (10, 10, 10))
            self._put_text(video, "Sensor preview disabled", 20, 30, (210, 210, 210))
        return np.hstack((panel, video))

    def _draw_panel(
        self,
        panel: np.ndarray,
        prediction: Any,
        probabilities: np.ndarray,
        aggregate_probabilities: np.ndarray,
    ) -> None:
        height, width = panel.shape[:2]
        margin = 14
        current_label = self._prediction_label(prediction, probabilities)
        current_confidence = float(np.max(probabilities))
        current_index = int(np.argmax(probabilities))
        current_accent = self._class_accent(current_index)

        self._draw_header(panel)
        hero_y = 48
        hero_h = 58
        self._draw_current_summary(
            panel,
            current_label,
            current_confidence,
            margin,
            hero_y,
            width - (2 * margin),
            hero_h,
            current_accent,
        )

        charts_y = hero_y + hero_h + 8
        status_h = 33
        charts_h = max(66, height - charts_y - status_h - 8)
        gap = 8
        chart_w = max(120, (width - (2 * margin) - gap) // 2)
        current_x = margin
        aggregate_x = current_x + chart_w + gap
        aggregate_w = max(120, width - margin - aggregate_x)
        aggregate_label = self._label_for_probabilities(aggregate_probabilities)
        aggregate_confidence = float(np.max(aggregate_probabilities))

        self._draw_chart_card(
            panel,
            "CURRENT",
            f"{current_confidence:.0%}",
            probabilities,
            current_x,
            charts_y,
            chart_w,
            charts_h,
            current_accent,
        )
        self._draw_chart_card(
            panel,
            f"AGG {len(self.probability_window)}/{self.aggregate_window_frames}",
            f"{aggregate_confidence:.0%} {self._display_label(aggregate_label)}",
            aggregate_probabilities,
            aggregate_x,
            charts_y,
            aggregate_w,
            charts_h,
            (120, 220, 120),
        )

        self._draw_status_strip(panel)

    def _new_panel(self, height: int, width: int) -> np.ndarray:
        top = np.array((18, 21, 28), dtype=np.float32)
        bottom = np.array((28, 39, 46), dtype=np.float32)

        mix = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        rows = ((top * (1.0 - mix)) + (bottom * mix)).astype(np.uint8)
        panel = np.repeat(rows[:, None, :], width, axis=1)

        cv2.circle(panel, (width - 18, 18), 62, (37, 58, 66), -1)
        cv2.circle(panel, (30, height - 12), 74, (23, 31, 38), -1)
        return panel

    def _draw_header(self, panel: np.ndarray) -> None:
        _, width = panel.shape[:2]
        x = 14
        self._put_text(panel, "DIGIT TACTILE", x, 18, (190, 230, 255), 0.43, 1)
        self._put_text(
            panel,
            "LIVE MATERIAL CLASSIFIER",
            x,
            34,
            (132, 150, 160),
            0.28,
            1,
        )

        badge_text = "LIVE"
        text_w = cv2.getTextSize(
            badge_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            1,
        )[0][0]
        pill_w = max(46, text_w + 28)
        pill_x = width - pill_w - x
        self._draw_rounded_rect(
            panel,
            pill_x,
            9,
            pill_w,
            18,
            (32, 62, 54),
            9,
        )
        cv2.circle(panel, (pill_x + 11, 18), 3, (120, 235, 160), -1)
        self._put_text(
            panel,
            badge_text,
            pill_x + 18,
            21,
            (195, 245, 210),
            0.25,
            1,
        )

        info = f"{self.camera_source or '-'}  |  {self.classifier.device}"
        self._put_text(
            panel,
            self._truncate(info, max(18, width // 9)),
            x,
            46,
            (175, 185, 190),
            0.27,
            1,
        )

    def _draw_current_summary(
        self,
        panel: np.ndarray,
        label: str,
        confidence: float,
        x: int,
        y: int,
        width: int,
        height: int,
        accent_color: tuple[int, int, int],
    ) -> None:
        self._draw_card(panel, x, y, width, height, (25, 31, 38))
        cv2.rectangle(panel, (x, y), (x + width - 1, y + 3), accent_color, -1)
        self._put_text(panel, "CURRENT TOUCH", x + 12, y + 18, (142, 156, 165), 0.29, 1)
        self._put_text_fit(
            panel,
            self._display_label(label).upper(),
            x + 12,
            y + 46,
            max(40, width - 96),
            (246, 248, 250),
            0.72,
            0.42,
            2,
            cv2.FONT_HERSHEY_DUPLEX,
        )
        self._draw_confidence_pill(
            panel,
            x + width - 76,
            y + 20,
            64,
            25,
            f"{confidence:.0%}",
            accent_color,
        )

    def _draw_chart_card(
        self,
        panel: np.ndarray,
        title: str,
        summary: str,
        probabilities: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        accent_color: tuple[int, int, int],
    ) -> None:
        self._draw_card(panel, x, y, width, height, (23, 29, 35))
        self._put_text(panel, title, x + 9, y + 15, (230, 235, 238), 0.31, 1)
        self._put_text(
            panel,
            self._truncate(summary, max(8, width // 9)),
            x + 9,
            y + 29,
            accent_color,
            0.27,
            1,
        )
        self._draw_probability_bars(
            panel,
            probabilities,
            x + 9,
            y + 31,
            width - 18,
            max(30, height - 35),
            accent_color,
        )

    def _draw_status_strip(self, panel: np.ndarray) -> None:
        height, width = panel.shape[:2]
        strip_h = 32
        y = max(0, height - strip_h)
        cv2.rectangle(panel, (0, y), (width, height), (12, 15, 19), -1)
        cv2.line(panel, (0, y), (width, y), (48, 57, 64), 1)

        status_color = self._status_color()
        cv2.circle(panel, (15, y + 11), 4, status_color, -1)
        status_text = self._truncate(self.status, max(16, (width - 34) // 7))
        self._put_text(panel, status_text, 25, y + 14, status_color, 0.29, 1)
        self._put_text(
            panel,
            "r reinit camera    q/esc quit",
            14,
            y + 28,
            (155, 165, 172),
            0.28,
            1,
        )

    def _draw_card(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        self._draw_rounded_rect(image, x, y, width, height, color, 9)
        cv2.rectangle(
            image,
            (x + 1, y + 1),
            (x + width - 2, y + height - 2),
            (54, 63, 70),
            1,
        )

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
        bar_height = max(3, min(8, row_height - 4))
        label_width = min(92, max(62, width // 2))
        bar_x = x + label_width + 4
        bar_width = max(16, width - label_width - 4)
        winner_index = int(np.argmax(probabilities))

        for index, class_name in enumerate(self.class_names):
            probability = float(np.clip(probabilities[index], 0.0, 1.0))
            row_y = y + (index * row_height)
            text_y = row_y + row_height - 1
            bar_y = row_y + max(2, (row_height - bar_height) // 2)
            is_winner = index == winner_index
            color = accent_color if is_winner else (84, 96, 104)
            label_color = (242, 245, 247) if is_winner else (170, 179, 184)

            if is_winner:
                self._draw_rounded_rect(
                    panel,
                    x - 4,
                    row_y,
                    width + 2,
                    row_height,
                    (32, 40, 47),
                    4,
                )

            self._put_text(
                panel,
                self._truncate(
                    self._display_label(class_name),
                    max(6, label_width // 7),
                ),
                x,
                text_y,
                label_color,
                0.25,
            )
            self._draw_rounded_rect(
                panel,
                bar_x,
                bar_y,
                bar_width,
                bar_height,
                (47, 55, 62),
                bar_height // 2,
            )
            fill_width = int(bar_width * probability)
            if fill_width > 0:
                self._draw_rounded_rect(
                    panel,
                    bar_x,
                    bar_y,
                    max(bar_height, fill_width),
                    bar_height,
                    color,
                    bar_height // 2,
                )
                if fill_width < bar_height:
                    cv2.rectangle(
                        panel,
                        (bar_x + fill_width, bar_y),
                        (bar_x + bar_height, bar_y + bar_height - 1),
                        (47, 55, 62),
                        -1,
                    )

        return y + (row_count * row_height)

    def _draw_confidence_pill(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        accent_color: tuple[int, int, int],
    ) -> None:
        self._draw_rounded_rect(image, x, y, width, height, (18, 23, 28), height // 2)
        cv2.rectangle(
            image,
            (x + 1, y + 1),
            (x + width - 2, y + height - 2),
            accent_color,
            1,
        )
        self._put_text_fit(
            image,
            text,
            x + 9,
            y + height - 8,
            max(10, width - 18),
            (245, 248, 250),
            0.42,
            0.28,
            1,
            cv2.FONT_HERSHEY_DUPLEX,
        )

    @staticmethod
    def _draw_rounded_rect(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        if width <= 0 or height <= 0:
            return

        x2 = x + width - 1
        y2 = y + height - 1
        radius = max(0, min(radius, width // 2, height // 2))
        if radius <= 1:
            cv2.rectangle(image, (x, y), (x2, y2), color, -1)
            return

        cv2.rectangle(image, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x, y + radius), (x2, y2 - radius), color, -1)
        cv2.circle(image, (x + radius, y + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y + radius), radius, color, -1)
        cv2.circle(image, (x + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)

    @staticmethod
    def _put_text_fit(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        max_width: int,
        color: tuple[int, int, int],
        scale: float,
        min_scale: float,
        thickness: int,
        font: int,
    ) -> None:
        text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if text_width > max_width:
            scale = max(min_scale, scale * (max_width / max(1, text_width)))
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _display_label(label: str) -> str:
        return str(label).replace("_", " ")

    @staticmethod
    def _class_accent(index: int) -> tuple[int, int, int]:
        palette = (
            (78, 203, 255),
            (255, 174, 83),
            (122, 224, 139),
            (232, 156, 255),
            (88, 206, 210),
            (116, 159, 255),
            (255, 144, 170),
        )
        return palette[index % len(palette)]

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
