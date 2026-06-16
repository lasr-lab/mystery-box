"""PySide6 desktop frontend for the DIGIT tactile classification demo."""

from __future__ import annotations

import glob
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import typing
from typing import Any, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# PySide6 exposes a typing backport marker as `typing.Self` on Python 3.9, which can
# make torchvision annotation handling fail if torchvision is imported afterwards.
if sys.version_info < (3, 11) and hasattr(typing, "Self"):
    del typing.Self

# Import the Torch/TorchVision inference stack before PySide6. With the current
# Python 3.9 + Torch 2.7 + TorchVision 0.22 stack, importing PySide6 first can
# make Torch's later ``typing.Self`` annotations fail during torchvision import.
from src.demo.inference import DemoClassifier

from PySide6.QtCore import (
    QObject,
    QMetaObject,
    Qt,
    QThread,
    QRectF,
    QSize,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
    QTransform,
)
from PySide6.QtSvg import QSvgRenderer

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


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
    is_sensor_frame: bool
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


class AspectRatioVideoStage(QWidget):
    """Centers the video label at the rotated DIGIT frame aspect ratio."""

    resized = Signal()

    def __init__(
        self,
        *,
        aspect_width: int,
        aspect_height: int,
        parent: Optional[QWidget],
    ) -> None:
        super().__init__(parent)
        self.aspect_width = aspect_width
        self.aspect_height = aspect_height
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("videoLabel")
        self.setMinimumSize(320, 240)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return

        label_height = height
        label_width = int(round(label_height * self.aspect_width / self.aspect_height))
        if label_width > width:
            label_width = width
            label_height = int(round(label_width * self.aspect_height / self.aspect_width))

        x = (width - label_width) // 2
        y = (height - label_height) // 2
        self.label.setGeometry(x, y, max(1, label_width), max(1, label_height))
        self.resized.emit()


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
        "attribution": "powered by LASR Lab @ TU Dresden",
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
        "aggregate": "Aggregate",
        "unavailable": "unavailable",
        "sensor_preview_disabled": "Sensor preview disabled",
        "reinitializing_camera": "Reinitializing camera",
    },
    "de": {
        "window_title": "Kannst du die KI schlagen?",
        "headline": "Kannst du die KI schlagen?",
        "instructions": (
            "Ber\u00fchre die Stoffe, die in der Box versteckt sind. Nimm dann den "
            "taktilen Sensor und pr\u00fcfe, ob die KI die Stoffe richtig "
            "klassifizieren kann."
        ),
        "attribution": "powered by LASR Lab @ TU Dresden",
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
        "aggregate": "Aggregiert",
        "unavailable": "nicht verf\u00fcgbar",
        "sensor_preview_disabled": "Sensorvorschau deaktiviert",
        "reinitializing_camera": "Kamera startet neu",
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

SECAI_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "secai-logo.svg"


def _normalize_language(value: Any) -> str:
    normalized = LANGUAGE_ALIASES[str(value).strip().lower()]
    if normalized not in UI_TEXT:
        raise ValueError(f"Unsupported UI language: {value!r}")
    return normalized


def _text(language: str, key: str) -> str:
    return UI_TEXT[language][key]


class CaptureInferenceWorker(QObject):
    """Owns DIGIT camera capture and model inference on a background Qt thread."""

    classifier_ready = Signal(object)
    frame_ready = Signal(object)
    status_changed = Signal(object)

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
        language: str,
    ) -> None:
        super().__init__()
        self.demo_cfg = demo_cfg
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
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
        self.probability_window: deque[np.ndarray] = deque(
            maxlen=self.aggregate_window_frames
        )
        self.timer: Optional[QTimer] = None
        self.status = "Starting demo."
        self.stopping = False

    @Slot()
    def start(self) -> None:
        """Load the model, open the camera, and start periodic capture."""

        self.classifier = DemoClassifier(
            str(self.demo_cfg["model_checkpoint"]),
            self.model_cfg,
            self.data_cfg,
            device=str(self.demo_cfg.get("device", "auto")),
        )
        self.class_names = [str(name) for name in self.classifier.class_names]
        if not self.class_names:
            raise RuntimeError("Demo classifier did not provide any class names.")

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
        sensor_fps = max(1, int(self.sensor_cfg.get("fps", 30)))
        ui_fps = min(sensor_fps, 15)
        self.timer.start(max(1, int(round(1000.0 / ui_fps))))

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
                is_sensor_frame=False,
                prediction_label=None,
                probabilities=None,
            )
        )

        device_path = self.sensor_cfg.get("device_path")
        if device_path:
            source = str(device_path)
        else:
            device_name = str(self.sensor_cfg["device_name"])
            matches: list[tuple[int, str]] = []
            for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
                name_path = Path(name_file)
                actual_name = name_path.read_text(encoding="utf-8").strip()
                if device_name.lower() not in actual_name.lower():
                    continue
                index = int(
                    (name_path.parent / "index").read_text(encoding="utf-8").strip()
                )
                matches.append((index, f"/dev/{name_path.parent.name}"))
            if not matches:
                raise RuntimeError(
                    f"No video device matching {device_name!r}. "
                    "Set demo.sensor.device_path=/dev/videoX."
                )
            source = sorted(matches)[0][1]

        backend_name = str(self.sensor_cfg.get("backend", "CAP_V4L2"))
        backend = getattr(cv2, backend_name)
        capture_source: Any = source
        if backend == cv2.CAP_V4L2:
            video_device = re.fullmatch(r"/dev/video(\d+)", source)
            if video_device is not None:
                capture_source = int(video_device.group(1))
            elif source.isdecimal():
                capture_source = int(source)

        camera = cv2.VideoCapture(capture_source, backend)
        fourcc = cv2.VideoWriter_fourcc(*str(self.sensor_cfg["fourcc"])[:4])
        camera.set(cv2.CAP_PROP_FOURCC, fourcc)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.sensor_cfg["width"]))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.sensor_cfg["height"]))
        camera.set(cv2.CAP_PROP_FPS, int(self.sensor_cfg["fps"]))

        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open DIGIT camera source {source}.")

        warmup_frames = int(self.sensor_cfg["warmup_frames"])
        read_success = warmup_frames <= 0
        for _ in range(warmup_frames):
            ok, _ = camera.read()
            read_success = read_success or ok
        if not read_success:
            camera.release()
            raise RuntimeError(
                f"Could not read warmup frames from DIGIT camera source {source}."
            )

        self.camera = camera
        self.camera_source = source
        self.probability_window.clear()
        fourcc_value = int(camera.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = (
            "unknown"
            if fourcc_value <= 0
            else "".join(
                chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
            ).strip()
        )
        self.status = (
            f"Camera {source}: "
            f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
            f"{camera.get(cv2.CAP_PROP_FPS):.1f} fps, "
            f"{fourcc_text}"
        )
        print(self.status)
        self.status_changed.emit(
            RuntimeStatus(
                status=self.status,
                camera_source=self.camera_source,
                device=str(self.classifier.device),
            )
        )

    @Slot()
    def update_frame(self) -> None:
        """Read one camera frame, run inference, and emit the display payload."""

        if self.classifier is None or self.stopping:
            return

        if self.camera is None:
            raise RuntimeError("DIGIT camera is not initialized.")

        ok, frame = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the DIGIT camera.")

        expected_width = int(self.sensor_cfg["width"])
        expected_height = int(self.sensor_cfg["height"])
        height, width = frame.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"Expected {expected_width}x{expected_height} from DIGIT, "
                f"got {width}x{height}."
            )

        prediction = self.classifier.predict(frame)
        probabilities = np.asarray(
            prediction.probabilities,
            dtype=np.float32,
        ).reshape(-1)
        if probabilities.shape[0] != len(self.class_names):
            raise RuntimeError(
                "Prediction probability count does not match class count: "
                f"{probabilities.shape[0]} != {len(self.class_names)}."
            )
        prediction_label = str(prediction.label)
        self.probability_window.append(probabilities)

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
                is_sensor_frame=self.show_sensor_preview,
                prediction_label=prediction_label,
                probabilities=probabilities,
            )
        )

    def _frame_result(
        self,
        *,
        frame_bgr: np.ndarray,
        is_sensor_frame: bool,
        prediction_label: Optional[str],
        probabilities: Optional[np.ndarray],
    ) -> FrameResult:
        aggregate_probabilities = (
            np.mean(np.stack(tuple(self.probability_window), axis=0), axis=0)
            if self.probability_window
            else np.zeros(len(self.class_names), dtype=np.float32)
        )
        return FrameResult(
            frame_bgr=frame_bgr,
            is_sensor_frame=is_sensor_frame,
            status=self.status,
            camera_source=self.camera_source,
            device=str(self.classifier.device) if self.classifier is not None else None,
            prediction_label=prediction_label,
            probabilities=probabilities,
            aggregate_probabilities=aggregate_probabilities,
            aggregate_count=len(self.probability_window),
        )

    def _release_camera(self) -> None:
        camera = self.camera
        self.camera = None
        self.camera_source = None
        if camera is not None:
            camera.release()

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


class DigitTactileQtWindow(QMainWindow):
    """Qt main window that only renders worker results."""

    reinitialize_requested = Signal()
    language_changed = Signal(str)

    def __init__(
        self,
        demo_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        data_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        gui_cfg = demo_cfg["gui"]
        self.language = _normalize_language(gui_cfg.get("language", "en"))
        configured_window_name = str(gui_cfg.get("window_name", "")).strip()
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
        self._reinitialize_pending = True
        self._worker_stopping = False

        self.resize(1120, 700)
        self._build_ui()

        self.worker_thread = QThread(self)
        self.worker = CaptureInferenceWorker(
            demo_cfg,
            model_cfg,
            data_cfg,
            self.language,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.start)
        self.worker.classifier_ready.connect(self._handle_classifier_ready)
        self.worker.frame_ready.connect(self._handle_frame_result)
        self.worker.status_changed.connect(self._handle_status)
        self.reinitialize_requested.connect(self.worker.reinitialize_camera)
        self.language_changed.connect(self.worker.set_language)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()

        QShortcut(QKeySequence("R"), self, activated=self._request_reinitialize)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence.StandardKey.Quit, self, activated=self.close)

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

        self.video_stage = AspectRatioVideoStage(
            aspect_width=3,
            aspect_height=4,
            parent=video_panel,
        )
        self.video_stage.resized.connect(self._resize_video_pixmap)
        self.video_label = self.video_stage.label
        video_layout.addWidget(self.video_stage, stretch=1)

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

        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        brand_layout = QVBoxLayout()
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(2)
        self.logo_label = QLabel(side_panel)
        self.logo_label.setObjectName("secaiLogo")
        self.logo_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        logo_rendered = False
        if QSvgRenderer is not None and SECAI_LOGO_PATH.exists():
            try:
                renderer = QSvgRenderer(str(SECAI_LOGO_PATH))
                if renderer.isValid():
                    intrinsic_size = renderer.defaultSize()
                    if (
                        intrinsic_size.isValid()
                        and intrinsic_size.width() > 0
                        and intrinsic_size.height() > 0
                    ):
                        target_width = 220
                        target_height = int(
                            intrinsic_size.height()
                            * (target_width / intrinsic_size.width())
                        )
                        target_size = QSize(target_width, max(1, target_height))
                        device_pixel_ratio = max(1.0, self.devicePixelRatioF())
                        pixmap_size = QSize(
                            int(round(target_size.width() * device_pixel_ratio)),
                            int(round(target_size.height() * device_pixel_ratio)),
                        )
                        logo_pixmap = QPixmap(pixmap_size)
                        logo_pixmap.setDevicePixelRatio(device_pixel_ratio)
                        logo_pixmap.fill(Qt.GlobalColor.transparent)
                        painter = QPainter(logo_pixmap)
                        painter.setRenderHints(
                            QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform
                        )
                        renderer.render(
                            painter,
                            QRectF(
                                0,
                                0,
                                target_size.width(),
                                target_size.height(),
                            ),
                        )
                        painter.end()
                        self.logo_label.setPixmap(logo_pixmap)
                        logo_rendered = True
            except Exception:
                logo_rendered = False

        if not logo_rendered:
            self.logo_label.setText("SECAI")
        self.logo_label.setMinimumHeight(48)
        brand_layout.addWidget(self.logo_label)
        header_layout.addLayout(brand_layout)

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
        header_layout.addLayout(language_layout)
        side_layout.addLayout(header_layout)

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
        self.attribution_label = QLabel(side_panel)
        self.attribution_label.setObjectName("attributionLabel")
        self.attribution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attribution_label.setWordWrap(True)
        side_layout.addWidget(self.attribution_label)

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
                background: #edf7fa;
                color: #17324f;
            }
            QFrame#videoPanel, QFrame#sidePanel {
                background: #ffffff;
                border: 1px solid #c9e7ef;
                border-radius: 14px;
            }
            QLabel {
                color: #17324f;
                font-size: 14px;
            }
            QLabel#secaiLogo {
                color: #305886;
                font-size: 30px;
                font-weight: 800;
            }
            QLabel#brandInfoLabel {
                color: #305886;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            QLabel#titleLabel {
                color: #305886;
                font-size: 27px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QLabel#instructionsLabel {
                color: #28465e;
                font-size: 16px;
                line-height: 1.35;
            }
            QLabel#languageLabel {
                color: #557086;
            }
            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 700;
                color: #305886;
            }
            QLabel#shortcutLabel {
                color: #557086;
            }
            QLabel#attributionLabel {
                color: #305886;
                font-size: 12px;
                font-weight: 700;
                padding-top: 8px;
            }
            QLabel#videoLabel {
                background: #06111f;
                color: #c9e7ef;
            }
            QPushButton {
                background: #305886;
                color: #ffffff;
                border: 0;
                border-radius: 8px;
                padding: 9px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #4eaec8;
                color: #17324f;
            }
            QPushButton:pressed {
                background: #00008c;
            }
            QToolButton#detailsToggle {
                background: #e8f7f8;
                color: #305886;
                border: 1px solid #b7dfe8;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 700;
            }
            QToolButton#detailsToggle:hover {
                background: #d8f0f4;
            }
            QComboBox {
                background: #f7fbfc;
                color: #305886;
                border: 1px solid #b7dfe8;
                border-radius: 8px;
                outline: 0;
                padding: 6px 12px;
                min-width: 104px;
            }
            QComboBox:hover, QComboBox:focus, QComboBox:on {
                background: #f7fbfc;
                color: #305886;
                border: 1px solid #b7dfe8;
                outline: 0;
            }
            QComboBox::drop-down {
                background: transparent;
                border: 0;
                width: 0px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #17324f;
                selection-background-color: #d8f0f4;
                selection-color: #17324f;
                border: 1px solid #b7dfe8;
                outline: 0;
            }
            QFrame#separator {
                background: #c9e7ef;
                max-height: 1px;
                border: 0;
            }
            QProgressBar {
                background: #e7f3f5;
                border: 0;
                border-radius: 5px;
                height: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #00aaaf;
                border-radius: 5px;
            }
            """
        )

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

        if self.last_pixmap is None:
            self.video_label.setText(self._tr("starting_camera"))

        self.title_label.setText(self._tr("headline"))
        self.instructions_label.setText(self._tr("instructions"))
        self.attribution_label.setText(self._tr("attribution"))
        self.language_label.setText(self._tr("language"))
        self.shortcuts_label.setText(self._tr("shortcuts"))
        self._refresh_language_combo()
        self._set_details_visible(self.details_toggle.isChecked())
        self._set_reinitialize_pending(self._reinitialize_pending)
        self._refresh_runtime_labels()
        for index, class_name in enumerate(self.class_names):
            if index < len(self.current_rows):
                self.current_rows[index].label.setText(
                    self._display_class_name(class_name)
                )
            if index < len(self.aggregate_rows):
                self.aggregate_rows[index].label.setText(
                    self._display_class_name(class_name)
                )
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
                f"{self._tr('aggregate')}: {self._tr('unavailable')}"
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
                f"{self._tr('aggregate')}: "
                f"{self._display_class_name(aggregate_label)} "
                f"({aggregate_confidence:.0%})"
            )
        else:
            self.aggregate_title.setText(
                f"{self._tr('aggregate')}: {self._tr('unavailable')}"
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
        return CLASS_DISPLAY_NAMES[self.language][class_name]

    def _display_status(self, status: str) -> str:
        if self.language == "en":
            return status

        if status == "Starting demo.":
            return self._tr("status_initial")
        if status == "Reinitializing camera...":
            return self._tr("reinitializing")
        if status.startswith("Camera "):
            return f"Kamera {status[len('Camera '):]}"
        return status

    @Slot()
    def _request_reinitialize(self) -> None:
        if self._reinitialize_pending or self._worker_stopping:
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
        self.reinitialize_button.setEnabled(not self._worker_stopping)

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
        self._set_video_frame(result.frame_bgr, rotate_left=result.is_sensor_frame)
        self._last_camera_source = result.camera_source
        if result.device is not None:
            self._last_device = result.device
        self._last_status = result.status
        self._refresh_runtime_labels()
        self._set_status_style(result.status)
        self._update_predictions(result)

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
                "font-weight: 700; color: #305886;"
                if index == winner_index and probability > 0.0
                else "font-weight: 400; color: #17324f;"
            )

    def _set_video_frame(self, frame_bgr: np.ndarray, *, rotate_left: bool) -> None:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        frame = np.ascontiguousarray(frame)
        height, width, _ = frame.shape
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        if rotate_left:
            image = image.transformed(
                QTransform().rotate(-90),
                Qt.TransformationMode.SmoothTransformation,
            )
        self.last_pixmap = QPixmap.fromImage(image)
        self._resize_video_pixmap()

    def _resize_video_pixmap(self) -> None:
        if self.last_pixmap is None:
            return
        target_size = self.video_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled_pixmap = self.last_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _set_status_style(self, status: str) -> None:
        status_lower = status.lower()
        if any(
            marker in status_lower
            for marker in ("failed", "unavailable", "could not")
        ):
            self.status_label.setStyleSheet("color: #b45309;")
        elif status.startswith("Camera"):
            self.status_label.setStyleSheet("color: #33b887;")
        else:
            self.status_label.setStyleSheet("color: #305886;")

    def _label_for_probabilities(self, probabilities: np.ndarray) -> str:
        return self.class_names[int(np.argmax(probabilities))]

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._resize_video_pixmap()

    def closeEvent(self, event: Any) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def _stop_worker(self) -> None:
        if self.worker_thread.isRunning():
            self._worker_stopping = True
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
    application = QApplication.instance() or QApplication([])
    window = DigitTactileQtWindow(
        demo_cfg=_config_section(cfg, "demo"),
        model_cfg=_config_section(cfg, "model"),
        data_cfg=_config_section(cfg, "data"),
    )
    # window.show()
    window.showMaximized()
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
