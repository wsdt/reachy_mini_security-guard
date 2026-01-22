"""Face recognition using OpenCV DNN (no CMake/dlib required).

Uses YuNet for face detection and SFace for face embeddings.
Both models are small ONNX files (~5MB total) downloaded from Hugging Face.
Works on resource-constrained devices like Raspberry Pi.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from huggingface_hub import hf_hub_download

from reachy_mini_security_guard.security.face_database import FaceDatabase


logger = logging.getLogger(__name__)

# Hugging Face model repository
# Uses opencv/opencv_zoo which hosts official OpenCV models
HF_REPO_ID = os.environ.get("FACE_MODEL_REPO", "opencv/opencv_zoo")

# Model filenames within the repository
YUNET_FILENAME = "face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface/face_recognition_sface_2021dec.onnx"


def _download_model_from_hf(filename: str, repo_id: str = HF_REPO_ID) -> Path:
    """Download a model from Hugging Face Hub.
    
    Args:
        filename: Path to file within the repository.
        repo_id: Hugging Face repository ID.
        
    Returns:
        Local path to the downloaded model.
    """
    logger.info("Downloading %s from %s ...", filename, repo_id)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        logger.info("Model cached at %s", local_path)
        return Path(local_path)
    except Exception as e:
        logger.error("Failed to download model %s: %s", filename, e)
        raise


def _ensure_models() -> tuple[Path, Path]:
    """Ensure face detection and recognition models are available.
    
    Downloads from Hugging Face Hub if not already cached.
    
    Returns:
        Tuple of (yunet_path, sface_path).
    """
    yunet_path = _download_model_from_hf(YUNET_FILENAME)
    sface_path = _download_model_from_hf(SFACE_FILENAME)
    
    return yunet_path, sface_path


@dataclass
class RecognizedFace:
    """Result of face recognition."""

    name: str  # Display name (or "unknown")
    face_id: str  # Database ID (or "unknown")
    confidence: float  # Match confidence (0-1, higher = better match)
    bbox: tuple[int, int, int, int]  # Bounding box (top, right, bottom, left)
    is_known: bool  # Whether face was recognized


class FaceRecognizer:
    """Face recognition using OpenCV DNN (YuNet + SFace).
    
    This implementation uses lightweight ONNX models that work on
    resource-constrained devices like Raspberry Pi without needing
    CMake or dlib compilation.
    """

    def __init__(
        self,
        database: FaceDatabase,
        confidence_threshold: float = 0.6,
        detection_score_threshold: float = 0.7,
    ):
        """Initialize the face recognizer.

        Args:
            database: FaceDatabase instance for known faces.
            confidence_threshold: Minimum confidence to consider a match (0-1).
                                 Higher values are stricter.
            detection_score_threshold: Minimum score for face detection (0-1).
        """
        self.database = database
        self.confidence_threshold = confidence_threshold
        self.detection_score_threshold = detection_score_threshold
        
        # Download/load models from Hugging Face (cached automatically)
        yunet_path, sface_path = _ensure_models()
        
        # Initialize face detector (YuNet)
        # Input size will be set dynamically based on image
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),  # Default size, will be updated per-image
            score_threshold=detection_score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        
        # Initialize face recognizer (SFace)
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(sface_path),
            "",
        )
        
        logger.info(
            "Face recognizer initialized (YuNet + SFace, threshold=%.2f)",
            confidence_threshold,
        )

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

    def _detect_faces(
        self,
        image: NDArray[np.uint8],
    ) -> list[NDArray[np.float32]]:
        """Detect faces in an image using YuNet.
        
        Args:
            image: BGR image (OpenCV format).
            
        Returns:
            List of face detections. Each detection is an array with:
            [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            where re=right eye, le=left eye, nt=nose tip, rcm/lcm=right/left corner mouth
        """
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        
        _, faces = self._detector.detect(image)
        
        if faces is None:
            return []
        
        return [face for face in faces if face[-1] >= self.detection_score_threshold]

    def _get_face_embedding(
        self,
        image: NDArray[np.uint8],
        face: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Get face embedding using SFace.
        
        Args:
            image: BGR image.
            face: Face detection from YuNet.
            
        Returns:
            128-dimensional face embedding.
        """
        # Align and crop face
        aligned_face = self._recognizer.alignCrop(image, face)
        
        # Get embedding
        embedding = self._recognizer.feature(aligned_face)
        
        return embedding.flatten()

    def _face_to_bbox(
        self,
        face: NDArray[np.float32],
    ) -> tuple[int, int, int, int]:
        """Convert YuNet face detection to bbox format (top, right, bottom, left).
        
        Args:
            face: YuNet detection [x, y, w, h, ...].
            
        Returns:
            Bounding box as (top, right, bottom, left).
        """
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        return (y, x + w, y + h, x)  # top, right, bottom, left

    def encode_faces_from_image(
        self,
        image: NDArray[np.uint8],
        max_faces: int = 10,
    ) -> list[tuple[NDArray[np.float32], tuple[int, int, int, int]]]:
        """Extract face encodings from an image.

        Args:
            image: BGR image (OpenCV format).
            max_faces: Maximum number of faces to encode.

        Returns:
            List of (encoding, bbox) tuples.
        """
        faces = self._detect_faces(image)
        
        if not faces:
            return []
        
        # Limit number of faces
        faces = faces[:max_faces]
        
        results = []
        for face in faces:
            try:
                embedding = self._get_face_embedding(image, face)
                bbox = self._face_to_bbox(face)
                results.append((embedding, bbox))
            except Exception as e:
                logger.warning("Failed to encode face: %s", e)
        
        return results

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

            # Find best match using cosine similarity
            best_score = -1.0
            best_idx = -1
            
            for i, known_emb in enumerate(self._known_embeddings):
                # SFace uses cosine similarity (higher = more similar)
                score = self._recognizer.match(
                    encoding.reshape(1, -1),
                    known_emb.reshape(1, -1).astype(np.float32),
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Convert score to confidence (SFace cosine is already 0-1)
            confidence = max(0.0, min(1.0, best_score))

            # Check if match is good enough
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
    ) -> tuple[list[NDArray[np.float32]], int, int]:
        """Encode faces from enrollment images.

        Args:
            images: List of BGR images.

        Returns:
            Tuple of (encodings, successful_count, failed_count).
        """
        encodings: list[NDArray[np.float32]] = []
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
