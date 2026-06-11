from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf


class DigitImageCollector:
    """Collect DIGIT frames in single-frame or continuous recording mode."""

    CAPTURE_MODES = {"continuous", "single_frame"}

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.sensor_cfg = cfg["sensor"]
        self.capture_cfg = cfg.get("capture", {})
        self.gui_cfg = cfg["gui"]
        self.labels = {str(key): value for key, value in cfg["labels"].items()}
        self.target_dir = Path(cfg["target_dir"])
        self.window_name = str(self.gui_cfg["window_name"])
        self.capture_mode = str(self.capture_cfg.get("mode", "continuous"))
        if self.capture_mode not in self.CAPTURE_MODES:
            valid_modes = ", ".join(sorted(self.CAPTURE_MODES))
            raise ValueError(
                f"Unknown capture mode {self.capture_mode!r}. Use one of: {valid_modes}."
            )

        self.camera: Optional[Any] = None
        self.camera_source: Optional[str] = None
        self.window_ready = False
        self.saved_count = 0
        self.recording_key: Optional[str] = None
        self.recording_saved_count = 0
        self.status = self._idle_status()

        self.target_dir.mkdir(parents=True, exist_ok=True)
        for label in self.labels.values():
            (self.target_dir / label["class_name"]).mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        self._open_camera()
        self._print_instructions()

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    self.status = "Frame read failed. Press r to reinit."
                    if self._handle_key(cv2.waitKey(self._wait_delay()) & 0xFF, None):
                        break
                    continue

                self._require_expected_size(frame)
                self._save_recording_frame(frame)
                preview = self._draw_preview(frame)
                self._ensure_window(preview)
                cv2.imshow(self.window_name, preview)

                if self._handle_key(cv2.waitKey(self._wait_delay()) & 0xFF, frame):
                    break
        finally:
            self._release_camera()
            cv2.destroyAllWindows()

    def _open_camera(self) -> None:
        self._release_camera()
        source = self._camera_source()
        backend = getattr(cv2, str(self.sensor_cfg["backend"]))
        camera = cv2.VideoCapture(source, backend)

        fourcc = cv2.VideoWriter_fourcc(*str(self.sensor_cfg["fourcc"])[:4])
        camera.set(cv2.CAP_PROP_FOURCC, fourcc)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.sensor_cfg["width"]))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.sensor_cfg["height"]))
        camera.set(cv2.CAP_PROP_FPS, int(self.sensor_cfg["fps"]))

        if not camera.isOpened():
            raise RuntimeError(f"Could not open DIGIT camera source {source}.")

        for _ in range(int(self.sensor_cfg["warmup_frames"])):
            camera.read()

        self.camera = camera
        self.camera_source = source
        self.window_ready = False
        self.recording_key = None
        self.recording_saved_count = 0
        self.status = (
            f"Camera {source}: "
            f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
            f"{camera.get(cv2.CAP_PROP_FPS):.1f} fps, "
            f"{self._fourcc_to_string(int(camera.get(cv2.CAP_PROP_FOURCC)))}"
        )
        print(self.status)

    def _camera_source(self) -> str:
        device_path = self.sensor_cfg.get("device_path")
        if device_path:
            return str(device_path)

        device_name = str(self.sensor_cfg["device_name"])
        source = self._find_video_device(device_name)
        if source is None:
            raise RuntimeError(
                f"No video device matching {device_name!r}. "
                "Set datacollection.sensor.device_path=/dev/videoX."
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
                    DigitImageCollector._device_index(name_path.parent),
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

    def _handle_key(self, key: int, frame: Optional[Any]) -> bool:
        if key == 255:
            return False
        if key in (27, ord("q"), ord("Q")):
            return True

        char = chr(key) if 0 <= key <= 255 else ""
        if char.lower() == "r":
            self._open_camera()
        elif char in self.labels and frame is not None:
            if self.capture_mode == "continuous":
                self._toggle_recording(char)
            else:
                self._save_frame(char, frame)
        return False

    def _toggle_recording(self, key: str) -> None:
        if self.recording_key == key:
            self._stop_recording()
        else:
            self._start_recording(key)

    def _start_recording(self, key: str) -> None:
        self.recording_key = key
        self.recording_saved_count = 0
        class_name = self._class_name(key)
        self.status = f"Recording {class_name}. Press {key} to stop."
        print(self.status)

    def _stop_recording(self) -> None:
        if self.recording_key is None:
            return

        class_name = self._class_name(self.recording_key)
        saved_count = self.recording_saved_count
        self.recording_key = None
        self.recording_saved_count = 0
        self.status = f"Stopped {class_name} after {saved_count} frame(s)."
        print(self.status)

    def _save_recording_frame(self, frame: Any) -> None:
        if self.recording_key is None:
            return

        class_name = self._class_name(self.recording_key)
        self._save_frame(self.recording_key, frame, announce=False)
        self.recording_saved_count += 1
        self.status = f"Recording {class_name}: saved {self.recording_saved_count} frame(s)."

    def _save_frame(self, key: str, frame: Any, *, announce: bool = True) -> None:
        label = self.labels[key]
        class_name = str(label["class_name"])
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
        output_path = self.target_dir / class_name / f"{timestamp}_{class_name}.png"

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write image to {output_path}.")

        self.saved_count += 1
        self.status = f"Saved {class_name}: {output_path.name}"
        if announce:
            print(self.status)

    def _class_name(self, key: str) -> str:
        return str(self.labels[key]["class_name"])

    def _require_expected_size(self, frame: Any) -> None:
        expected_width = int(self.sensor_cfg["width"])
        expected_height = int(self.sensor_cfg["height"])
        height, width = frame.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"Expected {expected_width}x{expected_height} from DIGIT, got {width}x{height}."
            )

    def _ensure_window(self, preview: Any) -> None:
        if self.window_ready:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        height, width = preview.shape[:2]
        scale = float(self.gui_cfg["display_scale"])
        cv2.resizeWindow(self.window_name, int(width * scale), int(height * scale))
        self.window_ready = True

    def _draw_preview(self, frame: Any) -> Any:
        panel_width = int(self.gui_cfg["side_panel_width"])
        preview = cv2.copyMakeBorder(
            frame,
            0,
            0,
            0,
            panel_width,
            cv2.BORDER_CONSTANT,
            value=(28, 28, 28),
        )
        self._draw_panel_text(preview, frame.shape[1], frame.shape[0])
        return preview

    def _draw_panel_text(self, preview: Any, frame_width: int, frame_height: int) -> None:
        x = frame_width + 12
        y = 18
        line_height = 15
        font_scale = 0.38
        lines = ["CLASSES"]
        lines += [
            f"{key}: {self.labels[key]['display_name']}"
            for key in sorted(self.labels, key=int)
        ]
        lines += [
            "",
            "CONTROLS",
            f"mode: {self.capture_mode}",
            self._class_key_help(),
            "r: reinit q/esc: quit",
            f"saved: {self.saved_count}",
            f"rec: {self._recording_class_name()}",
        ]

        for line in lines:
            if line:
                color = (
                    (170, 220, 255)
                    if line in {"CLASSES", "CONTROLS"}
                    else (245, 245, 245)
                )
                cv2.putText(
                    preview,
                    line,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            y += line_height

        cv2.putText(
            preview,
            self._short_status(),
            (x, frame_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self._status_color(),
            1,
            cv2.LINE_AA,
        )

    def _class_key_help(self) -> str:
        if self.capture_mode == "continuous":
            return "0-6: start/stop"
        return "0-6: save image"

    def _recording_class_name(self) -> str:
        if self.recording_key is None:
            return "off"
        return self._class_name(self.recording_key)

    def _status_color(self) -> tuple[int, int, int]:
        if self.status.startswith("Recording"):
            return (80, 190, 255)
        if self.status.startswith(("Saved", "Stopped")):
            return (120, 220, 120)
        return (210, 210, 210)

    def _short_status(self) -> str:
        max_chars = max(12, int(self.gui_cfg["side_panel_width"]) // 8)
        if len(self.status) <= max_chars:
            return self.status
        return f"{self.status[: max_chars - 3]}..."

    def _idle_status(self) -> str:
        return f"{self._class_key_help()}, r to reinit, q/esc to quit."

    def _wait_delay(self) -> int:
        return int(self.gui_cfg["wait_key_delay_ms"])

    def _print_instructions(self) -> None:
        print(f"Saving PNG captures to: {self.target_dir}")
        print(f"Capture mode: {self.capture_mode}")
        for key in sorted(self.labels, key=int):
            label = self.labels[key]
            print(f"  {key}: {label['class_name']} ({label['display_name']})")
        print(f"{self._class_key_help()}. Press r to reinitialize the sensor. Press q or esc to quit.")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    collector = DigitImageCollector(
        OmegaConf.to_container(cfg.datacollection, resolve=True)
    )
    collector.run()


if __name__ == "__main__":
    main()
