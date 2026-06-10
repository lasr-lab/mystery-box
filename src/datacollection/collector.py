from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf


class DigitImageCollector:
    """Collect one DIGIT frame per keypress."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.sensor_cfg = cfg["sensor"]
        self.gui_cfg = cfg["gui"]
        self.labels = {str(key): value for key, value in cfg["labels"].items()}
        self.target_dir = Path(cfg["target_dir"])
        self.window_name = str(self.gui_cfg["window_name"])
        self.camera: Any | None = None
        self.camera_source: str | None = None
        self.window_ready = False
        self.saved_count = 0
        self.status = "Press 0-6 to save, r to reinit, q/esc to quit."

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
    def _find_video_device(device_name: str) -> str | None:
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

    def _handle_key(self, key: int, frame: Any | None) -> bool:
        if key == 255:
            return False
        if key in (27, ord("q"), ord("Q")):
            return True

        char = chr(key) if 0 <= key <= 255 else ""
        if char.lower() == "r":
            self._open_camera()
        elif char in self.labels and frame is not None:
            self._save_frame(char, frame)
        return False

    def _save_frame(self, key: str, frame: Any) -> None:
        label = self.labels[key]
        class_name = str(label["class_name"])
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
        output_path = self.target_dir / class_name / f"{timestamp}_{class_name}.png"

        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to write image to {output_path}.")

        self.saved_count += 1
        self.status = f"Saved {class_name}: {output_path.name}"
        print(self.status)

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
            "r: reinit",
            "q/esc: quit",
            "",
            f"saved: {self.saved_count}",
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
            (120, 220, 120) if self.status.startswith("Saved") else (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    def _short_status(self) -> str:
        max_chars = max(12, int(self.gui_cfg["side_panel_width"]) // 8)
        if len(self.status) <= max_chars:
            return self.status
        return f"{self.status[: max_chars - 3]}..."

    def _wait_delay(self) -> int:
        return int(self.gui_cfg["wait_key_delay_ms"])

    def _print_instructions(self) -> None:
        print(f"Saving PNG captures to: {self.target_dir}")
        for key in sorted(self.labels, key=int):
            label = self.labels[key]
            print(f"  {key}: {label['class_name']} ({label['display_name']})")
        print("Press r to reinitialize the sensor. Press q or esc to quit.")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    collector = DigitImageCollector(
        OmegaConf.to_container(cfg.datacollection, resolve=True)
    )
    collector.run()


if __name__ == "__main__":
    main()
