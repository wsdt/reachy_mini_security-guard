"""Security monitoring loop for continuous face recognition.

Runs in a separate thread, monitors camera feed, and emits events
when faces are detected/recognized.
"""

import time
import logging
import threading
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

from reachy_mini_security_guard.security.config import SecurityConfig
from reachy_mini_security_guard.security.face_database import FaceDatabase
from reachy_mini_security_guard.security.face_recognizer import FaceRecognizer, RecognizedFace


logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Event emitted by the security monitor."""

    event_type: str  # "greeting", "alarm", "status"
    known_faces: list[str] = field(default_factory=list)  # Names of recognized faces
    unknown_count: int = 0  # Number of unknown faces
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""  # Human-readable message


class SecurityMonitor:
    """Monitors camera feed for face recognition.

    Runs in a separate thread and emits events via callbacks when:
    - Known faces are detected (for greeting)
    - Only unknown faces are detected (for alarm)
    """

    def __init__(
        self,
        camera_worker: Any,  # CameraWorker instance
        config: SecurityConfig,
        on_event: Optional[Callable[[SecurityEvent], None]] = None,
    ):
        """Initialize the security monitor.

        Args:
            camera_worker: CameraWorker instance for getting frames.
            config: SecurityConfig with settings.
            on_event: Callback function for security events.
        """
        self.camera_worker = camera_worker
        self.config = config
        self.on_event = on_event

        # Initialize face database and recognizer
        if config.face_db_path is None:
            raise ValueError("face_db_path must be set in config")

        self.database = FaceDatabase(config.face_db_path)
        self.recognizer = FaceRecognizer(
            self.database,
            confidence_threshold=config.confidence_threshold,
        )

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # State tracking
        self._armed = config.armed
        self._armed_lock = threading.Lock()

        # Cooldown tracking
        self._last_greeting_time: dict[str, float] = {}  # face_id -> timestamp
        self._last_alarm_time: float = 0.0

        # Recognition timing
        self._last_recognition_time: float = 0.0

    @property
    def armed(self) -> bool:
        """Get armed state (thread-safe)."""
        with self._armed_lock:
            return self._armed

    @armed.setter
    def armed(self, value: bool) -> None:
        """Set armed state (thread-safe)."""
        with self._armed_lock:
            self._armed = value
            self.config.armed = value
        logger.info("Security monitor %s", "ARMED" if value else "DISARMED")

    def update_config(self, config: SecurityConfig) -> None:
        """Update configuration settings.

        Args:
            config: New configuration.
        """
        self.config = config
        self.recognizer.set_confidence_threshold(config.confidence_threshold)
        with self._armed_lock:
            self._armed = config.armed
        logger.info("Security config updated")

    def start(self) -> None:
        """Start the monitoring loop in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Security monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Security monitor started")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Security monitor stopped")

    def _emit_event(self, event: SecurityEvent) -> None:
        """Emit a security event via callback."""
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error("Error in security event callback: %s", e)

    def _should_greet(self, face_id: str) -> bool:
        """Check if we should greet this face (cooldown check)."""
        now = time.time()
        last_time = self._last_greeting_time.get(face_id, 0.0)
        cooldown_seconds = self.config.greeting_cooldown_min * 60
        return (now - last_time) >= cooldown_seconds

    def _should_alarm(self) -> bool:
        """Check if we should trigger alarm (cooldown check)."""
        now = time.time()
        cooldown_seconds = self.config.unknown_cooldown_min * 60
        return (now - self._last_alarm_time) >= cooldown_seconds

    def _process_recognition_results(self, faces: list[RecognizedFace]) -> None:
        """Process recognition results and emit appropriate events.

        Args:
            faces: List of recognized faces.
        """
        if not faces:
            return

        now = time.time()

        # Separate known and unknown faces
        known_faces: list[RecognizedFace] = []
        unknown_faces: list[RecognizedFace] = []

        for face in faces:
            if face.is_known:
                known_faces.append(face)
            else:
                unknown_faces.append(face)

        # Handle greetings for known faces
        faces_to_greet: list[str] = []
        for face in known_faces:
            if self._should_greet(face.face_id):
                faces_to_greet.append(face.name)
                self._last_greeting_time[face.face_id] = now

        if faces_to_greet:
            event = SecurityEvent(
                event_type="greeting",
                known_faces=faces_to_greet,
                unknown_count=len(unknown_faces),
                message=f"Recognized: {', '.join(faces_to_greet)}",
            )
            self._emit_event(event)
            logger.info("Greeting event: %s", faces_to_greet)

        # Handle alarm for unknown faces (only if NO known faces present)
        if unknown_faces and not known_faces:
            if self._should_alarm():
                self._last_alarm_time = now
                count_text = "face" if len(unknown_faces) == 1 else "faces"
                event = SecurityEvent(
                    event_type="alarm",
                    known_faces=[],
                    unknown_count=len(unknown_faces),
                    message=f"Unknown {count_text} detected. Reporting to police.",
                )
                self._emit_event(event)
                logger.warning("ALARM: %d unknown %s detected", len(unknown_faces), count_text)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        logger.info("Security monitoring loop started")

        recognition_interval = 1.0 / self.config.recognition_hz

        while not self._stop_event.is_set():
            try:
                # Check if armed
                if not self.armed:
                    time.sleep(0.5)
                    continue

                # Check if camera is available
                if self.camera_worker is None:
                    time.sleep(1.0)
                    continue

                # Rate limit recognition
                now = time.time()
                time_since_last = now - self._last_recognition_time
                if time_since_last < recognition_interval:
                    time.sleep(recognition_interval - time_since_last)
                    continue

                # Get latest frame
                frame = self.camera_worker.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # Update timing
                self._last_recognition_time = time.time()

                # Run face recognition
                faces = self.recognizer.identify_faces(frame)

                # Process results
                self._process_recognition_results(faces)

            except Exception as e:
                logger.error("Error in security monitoring loop: %s", e)
                time.sleep(1.0)

        logger.info("Security monitoring loop stopped")

    def get_status(self) -> dict:
        """Get current security status.

        Returns:
            Status dictionary.
        """
        return {
            "armed": self.armed,
            "face_count": self.database.get_face_count(),
            "recognition_hz": self.config.recognition_hz,
            "unknown_cooldown_min": self.config.unknown_cooldown_min,
            "greeting_cooldown_min": self.config.greeting_cooldown_min,
            "confidence_threshold": self.config.confidence_threshold,
            "running": self._thread is not None and self._thread.is_alive(),
        }

    def enroll_face(
        self,
        name: str,
        images: list[tuple[str, bytes]],
    ) -> tuple[bool, str]:
        """Enroll a new face.

        Args:
            name: Name for the person.
            images: List of (filename, image_bytes) tuples.

        Returns:
            Tuple of (success, message).
        """
        import cv2
        import numpy as np

        if not images:
            return False, "No images provided"

        # Decode images
        decoded_images = []
        for filename, img_bytes in images:
            try:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded_images.append(img)
                else:
                    logger.warning("Failed to decode image: %s", filename)
            except Exception as e:
                logger.warning("Error decoding image %s: %s", filename, e)

        if not decoded_images:
            return False, "Could not decode any images"

        # Encode faces
        encodings, success_count, failed_count = self.recognizer.encode_enrollment_images(decoded_images)

        if not encodings:
            return False, f"No faces detected in any of the {len(decoded_images)} images"

        # Add to database
        success, message = self.database.add_face(name, encodings, images)

        if success:
            # Invalidate recognizer cache
            self.recognizer.invalidate_cache()

        return success, message

    def remove_face(self, name_or_id: str) -> tuple[bool, str]:
        """Remove a face from the database.

        Args:
            name_or_id: Name or ID of the face to remove.

        Returns:
            Tuple of (success, message).
        """
        success, message = self.database.remove_face(name_or_id)

        if success:
            # Invalidate recognizer cache
            self.recognizer.invalidate_cache()
            # Clear greeting cooldown for this face
            face_id = self.database._sanitize_id(name_or_id)
            self._last_greeting_time.pop(face_id, None)

        return success, message

    def list_faces(self) -> list[dict]:
        """List all enrolled faces.

        Returns:
            List of face info dictionaries.
        """
        return self.database.list_faces()
