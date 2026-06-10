from __future__ import annotations

import csv
import glob
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

import cv2


class DigitImageCollector:
    """Collect one DIGIT frame per keypress into class-specific directories."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.sensor_cfg = cfg["sensor"]
        self.gui_cfg = cfg["gui"]
        self.capture_cfg = cfg["capture"]
        self.labels = {str(key): value for key, value in cfg["labels"].items()}
        self.target_dir = Path(cfg["target_dir"])
        self.manifest_path = self.target_dir / cfg["manifest_filename"]
        self.window_name = str(self.gui_cfg["window_name"])
        self.camera: Any | None = None
        self.camera_source: int | str | None = None
        self.saved_count = 0
        self.last_status = "Press 0-6 to save, r to reinitialize, q or esc to quit."

        self.target_dir.mkdir(parents=True, exist_ok=True)
        for label in self.labels.values():
            (self.target_dir / label["class_name"]).mkdir(parents=True, exist_ok=True)
        if self.capture_cfg["save_manifest"]:
            self._ensure_manifest_header()

    def run(self) -> None:
        self._open_camera()
        self._print_instructions()

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    self.last_status = "Failed to read frame. Press r to reinitialize."
                    key = cv2.waitKey(int(self.gui_cfg["wait_key_delay_ms"])) & 0xFF
                    if self._handle_key(key, None):
                        break
                    continue

                frame = self._resize_if_needed(frame)
                preview = self._draw_overlay(frame) if self.gui_cfg["show_overlay"] else frame
                cv2.imshow(self.window_name, preview)

                key = cv2.waitKey(int(self.gui_cfg["wait_key_delay_ms"])) & 0xFF
                if self._handle_key(key, frame):
                    break
        finally:
            self._release_camera()
            cv2.destroyAllWindows()

    def _open_camera(self) -> None:
        self._release_camera()

        camera_source = self._resolve_camera_source()
        backend = self.sensor_cfg.get("backend")
        if backend:
            backend_id = getattr(cv2, str(backend))
            camera = cv2.VideoCapture(camera_source, backend_id)
        else:
            camera = cv2.VideoCapture(camera_source)

        fourcc = self.sensor_cfg.get("fourcc")
        if fourcc:
            fourcc_code = cv2.VideoWriter_fourcc(*str(fourcc)[:4])
            camera.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.sensor_cfg["width"]))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.sensor_cfg["height"]))
        camera.set(cv2.CAP_PROP_FPS, int(self.sensor_cfg["fps"]))

        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera source {camera_source}.")

        for _ in range(int(self.sensor_cfg.get("warmup_frames", 0))):
            camera.read()

        self.camera = camera
        self.camera_source = camera_source
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        self.last_status = (
            f"Camera initialized: source={camera_source}, "
            f"{actual_width}x{actual_height}@{actual_fps:.1f}fps"
        )
        print(self.last_status)

    def _resolve_camera_source(self) -> int | str:
        device_path = self.sensor_cfg.get("device_path")
        if device_path:
            return str(device_path)

        device_name = self.sensor_cfg.get("device_name")
        if device_name:
            matched_path = self._find_video_device_by_name(str(device_name))
            if matched_path is not None:
                return matched_path
            print(
                f"Could not find video device matching name {device_name!r}. "
                f"Falling back to camera_index={self.sensor_cfg['camera_index']}."
            )

        return int(self.sensor_cfg["camera_index"])

    def _find_video_device_by_name(self, expected_name: str) -> str | None:
        expected_name = expected_name.lower()
        candidates = sorted(glob.glob("/sys/class/video4linux/video*/name"))
        for name_file in candidates:
            name_path = Path(name_file)
            try:
                device_name = name_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue

            if expected_name not in device_name.lower():
                continue

            return f"/dev/{name_path.parent.name}"

        return None

    def _release_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _handle_key(self, key: int, frame: Any | None) -> bool:
        if key == 255:
            return False
        if key in (27, ord("q"), ord("Q")):
            return True

        char = chr(key) if 0 <= key <= 255 else ""
        if char.lower() == "r":
            self._open_camera()
            return False
        if char in self.labels:
            if frame is None:
                self.last_status = "No valid frame available to save."
            else:
                self._save_frame(char, frame)
            return False

        return False

    def _save_frame(self, key: str, frame: Any) -> None:
        label = self.labels[key]
        class_name = str(label["class_name"])
        class_dir = self.target_dir / class_name
        timestamp = datetime.now().astimezone()
        timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%S_%f")
        extension = str(self.capture_cfg["image_extension"]).lower().lstrip(".")
        filename = f"{timestamp_slug}_{class_name}.{extension}"
        output_path = class_dir / filename

        params = self._imwrite_params(extension)
        if not cv2.imwrite(str(output_path), frame, params):
            raise RuntimeError(f"Failed to write image to {output_path}.")

        self.saved_count += 1
        if self.capture_cfg["save_manifest"]:
            self._append_manifest(timestamp, key, label, output_path, frame)

        self.last_status = f"Saved {class_name}: {output_path.name}"
        print(self.last_status)

    def _imwrite_params(self, extension: str) -> list[int]:
        if extension in {"jpg", "jpeg"}:
            return [int(cv2.IMWRITE_JPEG_QUALITY), int(self.capture_cfg["jpg_quality"])]
        if extension == "png":
            return [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.capture_cfg["png_compression"])]
        return []

    def _resize_if_needed(self, frame: Any) -> Any:
        if not self.sensor_cfg.get("enforce_frame_size", True):
            return frame

        target_width = int(self.sensor_cfg["width"])
        target_height = int(self.sensor_cfg["height"])
        height, width = frame.shape[:2]
        if width == target_width and height == target_height:
            return frame
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _draw_overlay(self, frame: Any) -> Any:
        preview = frame.copy()
        lines = self._overlay_lines()
        line_height = 18
        overlay_height = min(preview.shape[0], 16 + line_height * len(lines))
        cv2.rectangle(preview, (0, 0), (preview.shape[1], overlay_height), (0, 0, 0), -1)

        y = 20
        for line in lines:
            color = (80, 220, 80) if line.startswith("Saved") else (255, 255, 255)
            cv2.putText(
                preview,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
            y += line_height
        return preview

    def _overlay_lines(self) -> list[str]:
        items = [
            f"{key}: {self.labels[key]['display_name']}"
            for key in sorted(self.labels, key=int)
        ]
        lines: list[str] = []
        current = ""
        for item in items:
            token = item if not current else f" | {item}"
            if len(current) + len(token) > 45:
                lines.append(current)
                current = item
            else:
                current += token
        if current:
            lines.append(current)
        lines.append("r: reinit sensor | q/esc: quit")
        lines.append(self.last_status)
        return lines

    def _ensure_manifest_header(self) -> None:
        if self.manifest_path.exists():
            return

        with self.manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "key",
                    "class_name",
                    "display_name",
                    "image_path",
                    "camera_source",
                    "configured_width",
                    "configured_height",
                    "configured_fps",
                    "frame_width",
                    "frame_height",
                ]
            )

    def _append_manifest(
        self,
        timestamp: datetime,
        key: str,
        label: dict[str, str],
        output_path: Path,
        frame: Any,
    ) -> None:
        height, width = frame.shape[:2]
        with self.manifest_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    timestamp.isoformat(),
                    key,
                    label["class_name"],
                    label["display_name"],
                    str(output_path),
                    self.camera_source,
                    self.sensor_cfg["width"],
                    self.sensor_cfg["height"],
                    self.sensor_cfg["fps"],
                    width,
                    height,
                ]
            )

    def _print_instructions(self) -> None:
        print(f"Saving captures to: {self.target_dir}")
        for key in sorted(self.labels, key=int):
            label = self.labels[key]
            print(f"  {key}: {label['class_name']} ({label['display_name']})")
        print("Press r to reinitialize the sensor. Press q or esc to quit.")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    collector_cfg = OmegaConf.to_container(cfg.datacollection, resolve=True)
    collector = DigitImageCollector(collector_cfg)
    collector.run()


if __name__ == "__main__":
    main()
