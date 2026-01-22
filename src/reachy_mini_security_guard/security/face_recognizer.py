"""Face recognition using the face_recognition library.

Provides face encoding and matching functionality.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from reachy_mini_security_guard.security.face_database import FaceDatabase


logger = logging.getLogger(__name__)

# Try to import face_recognition, provide helpful error if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning(
        "face_recognition library not available. "
        "Install with: pip install face_recognition "
        "(requires cmake and dlib)"
    )


@dataclass
class RecognizedFace:
    """Result of face recognition."""

    name: str  # Display name (or "unknown")
    face_id: str  # Database ID (or "unknown")
    confidence: float  # Match confidence (0-1, higher = better match)
    bbox: tuple[int, int, int, int]  # Bounding box (top, right, bottom, left)
    is_known: bool  # Whether face was recognized


class FaceRecognizer:
    """Face recognition using face_recognition library."""

    def __init__(
        self,
        database: FaceDatabase,
        confidence_threshold: float = 0.6,
    ):
        """Initialize the face recognizer.

        Args:
            database: FaceDatabase instance for known faces.
            confidence_threshold: Minimum confidence to consider a match (0-1).
                                 Lower values are stricter (require closer match).
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError(
                "face_recognition library is required. "
                "Install with: pip install face_recognition"
            )

        self.database = database
        self.confidence_threshold = confidence_threshold

        # Cache for known embeddings (refreshed when database changes)
        self._known_embeddings: NDArray[np.float64] | None = None
        self._known_face_ids: list[str] = []
        self._cache_valid = False

        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Refresh the known embeddings cache from database."""
        embeddings, face_ids = self.database.get_all_embeddings()
        self._known_embeddings = embeddings
        self._known_face_ids = face_ids
        self._cache_valid = True
        logger.debug(
            "Refreshed face cache: %d embeddings for %d faces",
            len(face_ids) if face_ids else 0,
            self.database.get_face_count(),
        )

    def invalidate_cache(self) -> None:
        """Mark cache as invalid (call after database changes)."""
        self._cache_valid = False

    def encode_faces_from_image(
        self,
        image: NDArray[np.uint8],
        max_faces: int = 10,
    ) -> list[tuple[NDArray[np.float64], tuple[int, int, int, int]]]:
        """Extract face encodings from an image.

        Args:
            image: BGR image (OpenCV format).
            max_faces: Maximum number of faces to encode.

        Returns:
            List of (encoding, bbox) tuples.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        # Convert BGR to RGB (face_recognition expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image, model="hog")

        if not face_locations:
            return []

        # Limit number of faces
        face_locations = face_locations[:max_faces]

        # Get encodings for each face
        encodings = face_recognition.face_encodings(rgb_image, face_locations)

        return list(zip(encodings, face_locations))

    def identify_faces(
        self,
        image: NDArray[np.uint8],
    ) -> list[RecognizedFace]:
        """Identify faces in an image.

        Args:
            image: BGR image (OpenCV format).

        Returns:
            List of RecognizedFace results.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        # Refresh cache if needed
        if not self._cache_valid:
            self._refresh_cache()

        # Get face encodings from image
        face_data = self.encode_faces_from_image(image)

        if not face_data:
            return []

        results: list[RecognizedFace] = []

        for encoding, bbox in face_data:
            # If no known faces, all are unknown
            if self._known_embeddings is None or len(self._known_embeddings) == 0:
                results.append(RecognizedFace(
                    name="Unknown",
                    face_id="unknown",
                    confidence=0.0,
                    bbox=bbox,
                    is_known=False,
                ))
                continue

            # Calculate face distances (lower = more similar)
            distances = face_recognition.face_distance(self._known_embeddings, encoding)

            # Find best match
            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]

            # Convert distance to confidence (0-1, higher = better)
            # face_recognition uses Euclidean distance, typical threshold is 0.6
            # distance of 0 = perfect match, distance of 1+ = very different
            confidence = max(0.0, 1.0 - best_distance)

            # Check if match is good enough
            # Note: confidence_threshold here means minimum confidence required
            # So we compare confidence >= threshold
            if confidence >= self.confidence_threshold:
                face_id = self._known_face_ids[best_idx]
                name = self.database.get_face_name(face_id)
                results.append(RecognizedFace(
                    name=name,
                    face_id=face_id,
                    confidence=confidence,
                    bbox=bbox,
                    is_known=True,
                ))
            else:
                results.append(RecognizedFace(
                    name="Unknown",
                    face_id="unknown",
                    confidence=confidence,
                    bbox=bbox,
                    is_known=False,
                ))

        return results

    def encode_enrollment_images(
        self,
        images: list[NDArray[np.uint8]],
    ) -> tuple[list[NDArray[np.float64]], int, int]:
        """Encode faces from enrollment images.

        Args:
            images: List of BGR images.

        Returns:
            Tuple of (encodings, successful_count, failed_count).
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return [], 0, len(images)

        encodings: list[NDArray[np.float64]] = []
        failed = 0

        for image in images:
            face_data = self.encode_faces_from_image(image, max_faces=1)

            if face_data:
                encodings.append(face_data[0][0])
            else:
                failed += 1
                logger.warning("No face found in enrollment image")

        return encodings, len(encodings), failed

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold.

        Args:
            threshold: New threshold (0-1).
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info("Set confidence threshold to %.2f", self.confidence_threshold)
