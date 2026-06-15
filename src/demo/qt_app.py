"""PySide6 desktop frontend for the DIGIT tactile classification demo."""

from __future__ import annotations

import glob
import re
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Import the Torch/TorchVision inference stack before PySide6. With the current
# Python 3.9 + Torch 2.7 + TorchVision 0.22 stack, importing PySide6 first can
# make Torch's later ``typing.Self`` annotations fail during torchvision import.
from src.demo.inference import DemoClassifier

try:
    from PySide6.QtCore import (
        QObject,
        QMetaObject,
        Qt,
        QThread,
        QTimer,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QProgressBar,
        QSizePolicy,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only in missing GUI envs.
    raise RuntimeError(
        "PySide6 is required for the Qt demo. Install the project dependencies "
        "before running `python -m src.demo.qt_app demo=default model=mobilevit_s`."
    ) from exc


@dataclass(frozen=True)
class ClassifierInfo:
    """Classifier metadata needed to build the probability panels."""

    class_names: list[str]
    device: str


@dataclass(frozen=True)
class RuntimeStatus:
    """Current worker status displayed by the Qt main thread."""

    status: str
    camera_source: Optional[str]
    device: Optional[str]


@dataclass(frozen=True)
class FrameResult:
    """One capture/inference update emitted from the worker thread."""

    frame_bgr: np.ndarray
    status: str
    camera_source: Optional[str]
    device: Optional[str]
    prediction_label: Optional[str]
    probabilities: Optional[np.ndarray]
    aggregate_probabilities: np.ndarray
    aggregate_count: int


@dataclass(frozen=True)
class ProbabilityRow:
    """Widgets for one class row in a probability panel."""

    label: QLabel
    bar: QProgressBar
    percent: QLabel


LANGUAGE_ALIASES = {
    "de": "de",
    "deutsch": "de",
    "german": "de",
    "en": "en",
    "english": "en",
}

UI_TEXT = {
    "en": {
        "window_title": "Can you beat the AI?",
        "headline": "Can you beat the AI?",
        "instructions": (
            "Touch the fabrics hidden in the box. Then take the tactile sensor "
            "and see if the AI can classify the fabrics correctly."
        ),
        "language": "Language:",
        "language_en": "English",
        "language_de": "Deutsch",
        "starting_camera": "Starting camera",
        "reinitialize": "Reinitialize camera",
        "reinitializing": "Reinitializing...",
        "shortcuts": "r: reinit  |  q/esc/ctrl+q: quit",
        "show_details": "Show details",
        "hide_details": "Hide details",
        "camera": "Camera",
        "inference": "Inference",
        "loading": "loading",
        "status_initial": "Starting demo.",
        "current": "Current",
        "unavailable": "unavailable",
        "sensor_preview_disabled": "Sensor preview disabled",
        "reinitializing_camera": "Reinitializing camera",
        "frame_unavailable": "Frame unavailable",
        "frame_unavailable_hint": (
            "Frame unavailable - press r or Reinitialize camera"
        ),
    },
    "de": {
        "window_title": "Kannst du die KI schlagen?",
        "headline": "Kannst du die KI schlagen?",
        "instructions": (
            "Ber\u00fchre die Stoffe, die in der Box versteckt sind. Nimm dann den "
            "taktilen Sensor und pr\u00fcfe, ob die KI die Stoffe richtig "
            "klassifizieren kann."
        ),
        "language": "Sprache:",
        "language_en": "Englisch",
        "language_de": "Deutsch",
        "starting_camera": "Kamera startet",
        "reinitialize": "Kamera neu starten",
        "reinitializing": "Kamera startet neu...",
        "shortcuts": "r: Kamera neu starten  |  q/esc/Strg+q: Beenden",
        "show_details": "Details anzeigen",
        "hide_details": "Details ausblenden",
        "camera": "Kamera",
        "inference": "Inferenz",
        "loading": "l\u00e4dt",
        "status_initial": "Demo startet.",
        "current": "Aktuell",
        "unavailable": "nicht verf\u00fcgbar",
        "sensor_preview_disabled": "Sensorvorschau deaktiviert",
        "reinitializing_camera": "Kamera startet neu",
        "frame_unavailable": "Bild nicht verf\u00fcgbar",
        "frame_unavailable_hint": (
            "Bild nicht verf\u00fcgbar - r dr\u00fccken oder Kamera neu starten"
        ),
    },
}

CLASS_DISPLAY_NAMES = {
    "en": {
        "nothing": "No contact",
        "cotton": "Cotton",
        "wool": "Wool",
        "curdory": "Corduroy",
        "synthetic_leather": "Synthetic leather",
        "teddy": "Teddy",
        "flower_fabric": "Flower fabric",
        "3dprint": "3D print",
        "finger": "Finger",
    },
    "de": {
        "nothing": "Kein Kontakt",
        "cotton": "Baumwolle",
        "wool": "Wolle",
        "curdory": "Kord",
        "synthetic_leather": "Kunstleder",
        "teddy": "Teddy",
        "flower_fabric": "Blumenstoff",
        "3dprint": "3D-Druck",
        "finger": "Finger",
    },
}


def _normalize_language(value: Any) -> str:
    return LANGUAGE_ALIASES.get(str(value).strip().lower(), "en")


def _text(language: str, key: str) -> str:
    return UI_TEXT.get(language, UI_TEXT["en"]).get(key, UI_TEXT["en"][key])


class CaptureInferenceWorker(QObject):
    """Owns DIGIT camera capture and model inference on a background Qt thread."""

    classifier_ready = Signal(object)
    frame_ready = Signal(object)
    status_changed = Signal(object)
    fatal_error = Signal(str)

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
        classifier_cls: Any,
        language: str,
    ) -> None:
        super().__init__()
        self.demo_cfg = demo_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.classifier_cls = classifier_cls
        self.language = _normalize_language(language)
        self.sensor_cfg = demo_cfg["sensor"]
        self.aggregate_window_frames = max(
            1, int(demo_cfg.get("aggregate_window_frames", 60))
        )
        self.show_sensor_preview = bool(demo_cfg.get("show_sensor_preview", True))

        self.classifier: Optional[Any] = None
        self.class_names: list[str] = []
        self.camera: Optional[Any] = None
        self.camera_source: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        self.probability_window: deque[np.ndarray] = deque(
            maxlen=self.aggregate_window_frames
        )
        self.timer: Optional[QTimer] = None
        self.status = "Starting demo."
        self.stopping = False

    @Slot()
    def start(self) -> None:
        """Load the model, open the camera, and start periodic capture."""

        try:
            self.classifier = self.classifier_cls(
                str(self.demo_cfg["model_checkpoint"]),
                self.model_cfg,
                self.data_cfg,
                device=str(self.demo_cfg.get("device", "auto")),
            )
            self.class_names = [str(name) for name in self.classifier.class_names]
            if not self.class_names:
                raise RuntimeError("Demo classifier did not provide any class names.")
        except Exception as exc:
            traceback.print_exc()
            self.fatal_error.emit(f"Classifier load failed: {exc}")
            return

        self.classifier_ready.emit(
            ClassifierInfo(
                class_names=list(self.class_names),
                device=str(self.classifier.device),
            )
        )
        self.reinitialize_camera()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self._timer_interval_ms())

    @Slot()
    def stop(self) -> None:
        """Stop the capture loop and release the camera in the worker thread."""

        self.stopping = True
        if self.timer is not None:
            self.timer.stop()
        self._release_camera()

    @Slot(str)
    def set_language(self, language: str) -> None:
        self.language = _normalize_language(language)

    @Slot()
    def reinitialize_camera(self) -> None:
        """Reopen the configured DIGIT camera source."""

        if self.classifier is None or self.stopping:
            return

        self.status = "Reinitializing camera..."
        self._release_camera()
        self.probability_window.clear()
        self.frame_ready.emit(
            self._frame_result(
                frame_bgr=self._placeholder_frame(
                    _text(self.language, "reinitializing_camera")
                ),
                prediction_label=None,
                probabilities=None,
            )
        )

        try:
            status = self._open_camera()
        except (AttributeError, OSError, RuntimeError, cv2.error) as exc:
            self._release_camera()
            status = f"Camera reinit failed: {exc}"

        self.status = status
        self.status_changed.emit(self._runtime_status())

    @Slot()
    def update_frame(self) -> None:
        """Read one camera frame, run inference, and emit the display payload."""

        if self.classifier is None or self.stopping:
            return

        frame = self._read_frame()
        if frame is None:
            self._release_camera()
            self.status = "Frame read failed. Press r or Reinitialize camera."
            self.probability_window.clear()
            self.frame_ready.emit(
                self._frame_result(
                    frame_bgr=self._error_frame(),
                    prediction_label=None,
                    probabilities=None,
                )
            )
            return

        try:
            self._require_expected_size(frame)
            self.last_frame = frame.copy()
            prediction = self.classifier.predict(frame)
            probabilities = self._prediction_probabilities(prediction)
            prediction_label = self._prediction_label(prediction, probabilities)
            self.probability_window.append(probabilities)
        except Exception as exc:
            self.status = f"Inference failed: {exc}"
            self.probability_window.clear()
            self.frame_ready.emit(
                self._frame_result(
                    frame_bgr=self._error_frame(),
                    prediction_label=None,
                    probabilities=None,
                )
            )
            return

        display_frame = (
            frame
            if self.show_sensor_preview
            else self._placeholder_frame(
                _text(self.language, "sensor_preview_disabled")
            )
        )
        self.frame_ready.emit(
            self._frame_result(
                frame_bgr=display_frame,
                prediction_label=prediction_label,
                probabilities=probabilities,
            )
        )

    def _frame_result(
        self,
        *,
        frame_bgr: np.ndarray,
        prediction_label: Optional[str],
        probabilities: Optional[np.ndarray],
    ) -> FrameResult:
        return FrameResult(
            frame_bgr=frame_bgr,
            status=self.status,
            camera_source=self.camera_source,
            device=str(self.classifier.device) if self.classifier is not None else None,
            prediction_label=prediction_label,
            probabilities=probabilities,
            aggregate_probabilities=self._aggregate_probabilities(),
            aggregate_count=len(self.probability_window),
        )

    def _runtime_status(self) -> RuntimeStatus:
        return RuntimeStatus(
            status=self.status,
            camera_source=self.camera_source,
            device=str(self.classifier.device) if self.classifier is not None else None,
        )

    def _open_camera(self) -> str:
        self._release_camera()

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
            self.camera_source = None
            raise

        self.camera = camera
        self.camera_source = source
        self.probability_window.clear()
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
                    CaptureInferenceWorker._device_index(name_path.parent),
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

    def _release_camera(self) -> None:
        camera = self.camera
        self.camera = None
        self.camera_source = None
        if camera is not None:
            camera.release()

    def _read_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            return None

        try:
            ok, frame = self.camera.read()
        except cv2.error:
            return None

        if not ok or frame is None:
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

    def _placeholder_frame(self, message: str) -> np.ndarray:
        width = int(self.sensor_cfg["width"])
        height = int(self.sensor_cfg["height"])
        frame = np.full((height, width, 3), (14, 16, 20), dtype=np.uint8)
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
            else self._placeholder_frame(_text(self.language, "frame_unavailable"))
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            frame,
            _text(self.language, "frame_unavailable_hint"),
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 190, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _prediction_label(self, prediction: Any, probabilities: np.ndarray) -> str:
        label = getattr(prediction, "label", None)
        if label:
            return str(label)
        return self._label_for_probabilities(probabilities)

    def _label_for_probabilities(self, probabilities: np.ndarray) -> str:
        return self.class_names[int(np.argmax(probabilities))]

    def _timer_interval_ms(self) -> int:
        sensor_fps = max(1, int(self.sensor_cfg.get("fps", 30)))
        ui_fps = min(sensor_fps, 15)
        return max(1, int(round(1000.0 / ui_fps)))


class DigitTactileQtWindow(QMainWindow):
    """Qt main window that only renders worker results."""

    reinitialize_requested = Signal()
    language_changed = Signal(str)

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
        classifier_cls: Any,
    ) -> None:
        super().__init__()
        self.demo_cfg = demo_cfg
        self.classifier_cls = classifier_cls
        self.language = self._initial_language()
        configured_window_name = str(
            demo_cfg.get("gui", {}).get("window_name", "")
        ).strip()
        self._custom_window_name = configured_window_name
        self._use_translated_window_title = (
            not configured_window_name
            or configured_window_name == "DIGIT Tactile Demo"
        )
        self.aggregate_window_frames = max(
            1, int(demo_cfg.get("aggregate_window_frames", 60))
        )
        self.class_names: list[str] = []
        self.current_rows: list[ProbabilityRow] = []
        self.aggregate_rows: list[ProbabilityRow] = []
        self.last_pixmap: Optional[QPixmap] = None
        self._last_camera_source: Optional[str] = None
        self._last_device: Optional[str] = None
        self._last_status = "Starting demo."
        self._last_current_label: Optional[str] = None
        self._last_current_probabilities: Optional[np.ndarray] = None
        self._last_aggregate_probabilities: Optional[np.ndarray] = None
        self._last_aggregate_count = 0
        self._fatal_error_seen = False
        self._reinitialize_pending = True
        self._worker_stopping = False

        self.resize(1120, 700)
        self._build_ui()
        self._setup_worker(demo_cfg, model_cfg, data_cfg)
        self._setup_shortcuts()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        video_panel = QFrame(root)
        video_panel.setObjectName("videoPanel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setSpacing(10)

        self.video_label = QLabel(video_panel)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.video_label.setObjectName("videoLabel")
        video_layout.addWidget(self.video_label, stretch=1)

        controls_layout = QHBoxLayout()
        self.reinitialize_button = QPushButton(video_panel)
        self.reinitialize_button.clicked.connect(self._request_reinitialize)
        controls_layout.addWidget(self.reinitialize_button)
        controls_layout.addStretch(1)
        self.shortcuts_label = QLabel(video_panel)
        self.shortcuts_label.setObjectName("shortcutLabel")
        controls_layout.addWidget(self.shortcuts_label)
        video_layout.addLayout(controls_layout)
        root_layout.addWidget(video_panel, stretch=3)

        side_panel = QFrame(root)
        side_panel.setObjectName("sidePanel")
        side_panel.setMinimumWidth(390)
        side_panel.setMaximumWidth(480)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)

        language_layout = QHBoxLayout()
        language_layout.addStretch(1)
        self.language_label = QLabel(side_panel)
        self.language_label.setObjectName("languageLabel")
        self.language_combo = QComboBox(side_panel)
        self._refresh_language_combo()
        self.language_combo.currentIndexChanged.connect(
            self._handle_language_changed
        )
        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.language_combo)
        side_layout.addLayout(language_layout)

        self.title_label = QLabel(side_panel)
        self.title_label.setObjectName("titleLabel")
        self.title_label.setWordWrap(True)
        side_layout.addWidget(self.title_label)

        self.instructions_label = QLabel(side_panel)
        self.instructions_label.setObjectName("instructionsLabel")
        self.instructions_label.setWordWrap(True)
        side_layout.addWidget(self.instructions_label)

        self.details_toggle = QToolButton(side_panel)
        self.details_toggle.setObjectName("detailsToggle")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.details_toggle.toggled.connect(self._set_details_visible)
        side_layout.addWidget(self.details_toggle)

        self.details_content = QWidget(side_panel)
        details_layout = QVBoxLayout(self.details_content)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)

        self.camera_label = QLabel(side_panel)
        self.inference_label = QLabel(side_panel)
        self.status_label = QLabel(side_panel)
        self.status_label.setWordWrap(True)
        details_layout.addWidget(self.camera_label)
        details_layout.addWidget(self.inference_label)
        details_layout.addWidget(self.status_label)

        details_layout.addWidget(self._separator(self.details_content))

        self.current_title = QLabel(side_panel)
        self.current_title.setObjectName("sectionTitle")
        details_layout.addWidget(self.current_title)
        self.current_rows_container = QWidget(self.details_content)
        self.current_rows_layout = QVBoxLayout(self.current_rows_container)
        self.current_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.current_rows_layout.setSpacing(6)
        details_layout.addWidget(self.current_rows_container)

        side_layout.addWidget(self.details_content)

        side_layout.addWidget(self._separator(side_panel))

        self.aggregate_title = QLabel(side_panel)
        self.aggregate_title.setObjectName("sectionTitle")
        side_layout.addWidget(self.aggregate_title)
        self.aggregate_rows_container = QWidget(side_panel)
        self.aggregate_rows_layout = QVBoxLayout(self.aggregate_rows_container)
        self.aggregate_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.aggregate_rows_layout.setSpacing(6)
        side_layout.addWidget(self.aggregate_rows_container)
        side_layout.addStretch(1)

        root_layout.addWidget(side_panel, stretch=1)
        self._apply_styles()
        self._apply_translations()
        self._set_details_visible(False)
        self._set_reinitialize_pending(True)
        self._set_status_style(self._last_status)

    def _separator(self, parent: QWidget) -> QFrame:
        separator = QFrame(parent)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setObjectName("separator")
        return separator

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #101820;
                color: #f4efe6;
            }
            QFrame#videoPanel, QFrame#sidePanel {
                background: #17242d;
                border: 1px solid #2e4654;
                border-radius: 14px;
            }
            QLabel {
                color: #f4efe6;
                font-size: 14px;
            }
            QLabel#titleLabel {
                color: #f8c765;
                font-size: 27px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QLabel#instructionsLabel {
                color: #dce8ef;
                font-size: 16px;
                line-height: 1.35;
            }
            QLabel#languageLabel {
                color: #9fb3bf;
            }
            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#shortcutLabel {
                color: #9fb3bf;
            }
            QLabel#videoLabel {
                background: #070a0d;
                border-radius: 10px;
                color: #9fb3bf;
            }
            QPushButton {
                background: #f8c765;
                color: #101820;
                border: 0;
                border-radius: 8px;
                padding: 9px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ffd989;
            }
            QPushButton:pressed {
                background: #dba84b;
            }
            QToolButton#detailsToggle {
                background: #243642;
                color: #f4efe6;
                border: 1px solid #3a5969;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 700;
            }
            QToolButton#detailsToggle:hover {
                background: #2b4350;
            }
            QComboBox {
                background: #243642;
                color: #f4efe6;
                border: 1px solid #3a5969;
                border-radius: 8px;
                padding: 6px 10px;
                min-width: 104px;
            }
            QComboBox QAbstractItemView {
                background: #17242d;
                color: #f4efe6;
                selection-background-color: #2e4654;
            }
            QFrame#separator {
                background: #2e4654;
                max-height: 1px;
                border: 0;
            }
            QProgressBar {
                background: #243642;
                border: 0;
                border-radius: 5px;
                height: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #5ec0d4;
                border-radius: 5px;
            }
            """
        )

    def _initial_language(self) -> str:
        gui_cfg = self.demo_cfg.get("gui", {})
        if not isinstance(gui_cfg, dict):
            return "en"
        return _normalize_language(gui_cfg.get("language", "en"))

    def _tr(self, key: str) -> str:
        return _text(self.language, key)

    def _refresh_language_combo(self) -> None:
        was_blocked = self.language_combo.blockSignals(True)
        self.language_combo.clear()
        self.language_combo.addItem(self._tr("language_en"), "en")
        self.language_combo.addItem(self._tr("language_de"), "de")
        index = self.language_combo.findData(self.language)
        self.language_combo.setCurrentIndex(max(index, 0))
        self.language_combo.blockSignals(was_blocked)

    def _apply_translations(self) -> None:
        if self._use_translated_window_title:
            self.setWindowTitle(self._tr("window_title"))
        else:
            self.setWindowTitle(self._custom_window_name)

        if self._fatal_error_seen:
            self.video_label.setText(self._display_status(self._last_status))
        elif self.last_pixmap is None:
            self.video_label.setText(self._tr("starting_camera"))

        self.title_label.setText(self._tr("headline"))
        self.instructions_label.setText(self._tr("instructions"))
        self.language_label.setText(self._tr("language"))
        self.shortcuts_label.setText(self._tr("shortcuts"))
        self._refresh_language_combo()
        self._set_details_visible(self.details_toggle.isChecked())
        self._set_reinitialize_pending(self._reinitialize_pending)
        self._refresh_runtime_labels()
        self._refresh_probability_row_labels()
        self._refresh_prediction_titles()

    def _refresh_runtime_labels(self) -> None:
        self.camera_label.setText(
            f"{self._tr('camera')}: {self._last_camera_source or '-'}"
        )
        device = (
            self._last_device
            if self._last_device is not None
            else self._tr("loading")
        )
        self.inference_label.setText(f"{self._tr('inference')}: {device}")
        self.status_label.setText(self._display_status(self._last_status))

    def _refresh_prediction_titles(self) -> None:
        if not self.class_names:
            self.current_title.setText(
                f"{self._tr('current')}: {self._tr('unavailable')}"
            )
            self.aggregate_title.setText(
                f"0/{self.aggregate_window_frames}: {self._tr('unavailable')}"
            )
            return

        if self._last_current_probabilities is None:
            self.current_title.setText(
                f"{self._tr('current')}: {self._tr('unavailable')}"
            )
        else:
            confidence = float(np.max(self._last_current_probabilities))
            raw_label = self._last_current_label or self._label_for_probabilities(
                self._last_current_probabilities
            )
            self.current_title.setText(
                f"{self._tr('current')}: {self._display_class_name(raw_label)} "
                f"({confidence:.0%})"
            )

        aggregate_probabilities = self._last_aggregate_probabilities
        if aggregate_probabilities is not None and self._last_aggregate_count > 0:
            aggregate_label = self._label_for_probabilities(aggregate_probabilities)
            aggregate_confidence = float(np.max(aggregate_probabilities))
            self.aggregate_title.setText(
                f"{self._last_aggregate_count}/{self.aggregate_window_frames}: "
                f"{self._display_class_name(aggregate_label)} "
                f"({aggregate_confidence:.0%})"
            )
        else:
            self.aggregate_title.setText(
                f"0/{self.aggregate_window_frames}: {self._tr('unavailable')}"
            )

    def _refresh_probability_row_labels(self) -> None:
        for index, class_name in enumerate(self.class_names):
            if index < len(self.current_rows):
                self.current_rows[index].label.setText(
                    self._display_class_name(class_name)
                )
            if index < len(self.aggregate_rows):
                self.aggregate_rows[index].label.setText(
                    self._display_class_name(class_name)
                )

    @Slot(bool)
    def _set_details_visible(self, visible: bool) -> None:
        self.details_content.setVisible(visible)
        self.details_toggle.setChecked(visible)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self.details_toggle.setText(
            self._tr("hide_details") if visible else self._tr("show_details")
        )

    @Slot(int)
    def _handle_language_changed(self, index: int) -> None:
        language = _normalize_language(self.language_combo.itemData(index))
        if language == self.language:
            return

        self.language = language
        self._apply_translations()
        self.language_changed.emit(language)

    def _display_class_name(self, class_name: str) -> str:
        display_names = CLASS_DISPLAY_NAMES.get(
            self.language,
            CLASS_DISPLAY_NAMES["en"],
        )
        if class_name in display_names:
            return display_names[class_name]
        humanized = class_name.replace("_", " ")
        return humanized[:1].upper() + humanized[1:]

    def _display_status(self, status: str) -> str:
        if self.language == "en":
            return status

        if status == "Starting demo.":
            return self._tr("status_initial")
        if status == "Reinitializing camera...":
            return self._tr("reinitializing")
        if status.startswith("Camera reinit failed:"):
            return f"Kamera-Neustart fehlgeschlagen:{status.split(':', 1)[1]}"
        if status.startswith("Frame read failed."):
            return (
                "Bild konnte nicht gelesen werden. Dr\u00fccke r oder starte die "
                "Kamera neu."
            )
        if status.startswith("Inference failed:"):
            return f"Inferenz fehlgeschlagen:{status.split(':', 1)[1]}"
        if status.startswith("Classifier load failed:"):
            return (
                "Klassifikator konnte nicht geladen werden:"
                f"{status.split(':', 1)[1]}"
            )
        if status.startswith("Camera "):
            return f"Kamera {status[len('Camera '):]}"
        return status

    def _setup_worker(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        self.worker_thread = QThread(self)
        self.worker = CaptureInferenceWorker(
            demo_cfg,
            model_cfg,
            data_cfg,
            self.classifier_cls,
            self.language,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.start)
        self.worker.classifier_ready.connect(self._handle_classifier_ready)
        self.worker.frame_ready.connect(self._handle_frame_result)
        self.worker.status_changed.connect(self._handle_status)
        self.worker.fatal_error.connect(self._handle_fatal_error)
        self.reinitialize_requested.connect(self.worker.reinitialize_camera)
        self.language_changed.connect(self.worker.set_language)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("R"), self, activated=self._request_reinitialize)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence.StandardKey.Quit, self, activated=self.close)

    @Slot()
    def _request_reinitialize(self) -> None:
        if (
            self._reinitialize_pending
            or self._worker_stopping
            or self._fatal_error_seen
        ):
            return

        self._set_reinitialize_pending(True)
        self.reinitialize_requested.emit()

    def _set_reinitialize_pending(self, pending: bool) -> None:
        self._reinitialize_pending = pending
        if pending:
            self.reinitialize_button.setText(self._tr("reinitializing"))
            self.reinitialize_button.setEnabled(False)
            return

        self.reinitialize_button.setText(self._tr("reinitialize"))
        self.reinitialize_button.setEnabled(
            not self._worker_stopping and not self._fatal_error_seen
        )

    @Slot(object)
    def _handle_classifier_ready(self, info: ClassifierInfo) -> None:
        self.class_names = list(info.class_names)
        self._last_device = info.device
        self._refresh_runtime_labels()
        self._rebuild_probability_rows()

    @Slot(object)
    def _handle_status(self, status: RuntimeStatus) -> None:
        self._set_reinitialize_pending(False)
        self._last_camera_source = status.camera_source
        if status.device is not None:
            self._last_device = status.device
        self._last_status = status.status
        self._refresh_runtime_labels()
        self._set_status_style(status.status)

    @Slot(object)
    def _handle_frame_result(self, result: FrameResult) -> None:
        self._set_video_frame(result.frame_bgr)
        self._last_camera_source = result.camera_source
        if result.device is not None:
            self._last_device = result.device
        self._last_status = result.status
        self._refresh_runtime_labels()
        self._set_status_style(result.status)
        self._update_predictions(result)

    @Slot(str)
    def _handle_fatal_error(self, message: str) -> None:
        self._fatal_error_seen = True
        self._set_reinitialize_pending(False)
        self._last_status = message
        self._refresh_runtime_labels()
        self._set_status_style(message)
        self.video_label.setText(self._display_status(message))

    def _rebuild_probability_rows(self) -> None:
        self._clear_layout(self.current_rows_layout)
        self._clear_layout(self.aggregate_rows_layout)
        self.current_rows = self._build_probability_rows(self.current_rows_layout)
        self.aggregate_rows = self._build_probability_rows(self.aggregate_rows_layout)
        zeros = np.zeros(len(self.class_names), dtype=np.float32)
        self._update_probability_rows(self.current_rows, zeros)
        self._update_probability_rows(self.aggregate_rows, zeros)
        self._refresh_prediction_titles()

    def _build_probability_rows(self, layout: QVBoxLayout) -> list[ProbabilityRow]:
        rows: list[ProbabilityRow] = []
        for class_name in self.class_names:
            row_widget = QWidget(self)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            label = QLabel(self._display_class_name(class_name), row_widget)
            label.setMinimumWidth(130)
            label.setObjectName("classLabel")
            bar = QProgressBar(row_widget)
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            percent = QLabel("0%", row_widget)
            percent.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            percent.setMinimumWidth(44)

            row_layout.addWidget(label)
            row_layout.addWidget(bar, stretch=1)
            row_layout.addWidget(percent)
            layout.addWidget(row_widget)
            rows.append(ProbabilityRow(label=label, bar=bar, percent=percent))
        return rows

    def _update_predictions(self, result: FrameResult) -> None:
        if not self.class_names:
            return

        self._last_current_label = result.prediction_label
        self._last_current_probabilities = result.probabilities
        self._last_aggregate_probabilities = result.aggregate_probabilities
        self._last_aggregate_count = result.aggregate_count

        if result.probabilities is None:
            current_probabilities = np.zeros(len(self.class_names), dtype=np.float32)
        else:
            current_probabilities = result.probabilities
        self._update_probability_rows(self.current_rows, current_probabilities)

        aggregate_probabilities = result.aggregate_probabilities
        self._update_probability_rows(self.aggregate_rows, aggregate_probabilities)
        self._refresh_prediction_titles()

    def _update_probability_rows(
        self,
        rows: list[ProbabilityRow],
        probabilities: np.ndarray,
    ) -> None:
        if not rows:
            return

        winner_index = int(np.argmax(probabilities)) if len(probabilities) else 0
        for index, row in enumerate(rows):
            probability = float(np.clip(probabilities[index], 0.0, 1.0))
            row.bar.setValue(int(round(probability * 1000)))
            row.percent.setText(f"{probability:.0%}")
            row.label.setStyleSheet(
                "font-weight: 700; color: #f8c765;"
                if index == winner_index and probability > 0.0
                else "font-weight: 400; color: #f4efe6;"
            )

    def _set_video_frame(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.video_label.setText(f"Invalid frame shape: {frame.shape}")
            return

        image_format = self._qimage_format("Format_BGR888")
        if image_format is None:
            frame = np.ascontiguousarray(frame[:, :, ::-1])
            image_format = self._qimage_format("Format_RGB888")
        else:
            frame = np.ascontiguousarray(frame)

        height, width, _ = frame.shape
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            image_format,
        ).copy()
        self.last_pixmap = QPixmap.fromImage(image)
        self._resize_video_pixmap()

    @staticmethod
    def _qimage_format(name: str) -> Optional[Any]:
        direct_format = getattr(QImage, name, None)
        if direct_format is not None:
            return direct_format
        return getattr(QImage.Format, name, None)

    def _resize_video_pixmap(self) -> None:
        if self.last_pixmap is None:
            return
        self.video_label.setPixmap(
            self.last_pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_status_style(self, status: str) -> None:
        status_lower = status.lower()
        if any(
            marker in status_lower
            for marker in ("failed", "unavailable", "could not")
        ):
            self.status_label.setStyleSheet("color: #ffb36b;")
        elif status.startswith("Camera"):
            self.status_label.setStyleSheet("color: #90d585;")
        else:
            self.status_label.setStyleSheet("color: #d2dde4;")

    def _label_for_probabilities(self, probabilities: np.ndarray) -> str:
        return self.class_names[int(np.argmax(probabilities))]

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._resize_video_pixmap()

    def closeEvent(self, event: Any) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def _stop_worker(self) -> None:
        if not hasattr(self, "worker_thread"):
            return
        if self.worker_thread.isRunning():
            self._worker_stopping = True
            if hasattr(self, "reinitialize_button"):
                self.reinitialize_button.setEnabled(False)
            QMetaObject.invokeMethod(
                self.worker,
                "stop",
                Qt.ConnectionType.QueuedConnection,
            )
            self.worker_thread.quit()
            if not self.worker_thread.wait(2000):
                print("Warning: Qt worker thread did not stop within 2000 ms.")

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class DigitTactileQtApp:
    """Small wrapper matching the existing demo entry point shape."""

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        self.demo_cfg = demo_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

    def run(self) -> None:
        classifier_cls = self._load_classifier_class()
        application = QApplication.instance() or QApplication([])
        window = DigitTactileQtWindow(
            demo_cfg=self.demo_cfg,
            model_cfg=self.model_cfg,
            data_cfg=self.data_cfg,
            classifier_cls=classifier_cls,
        )
        window.show()
        raise SystemExit(application.exec())

    @staticmethod
    def _load_classifier_class() -> Any:
        """Return the preloaded classifier class for the worker thread."""
        return DemoClassifier


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
    app = DigitTactileQtApp(
        demo_cfg=_config_section(cfg, "demo"),
        model_cfg=_config_section(cfg, "model"),
        data_cfg=_config_section(cfg, "data"),
    )
    app.run()


if __name__ == "__main__":
    main()
