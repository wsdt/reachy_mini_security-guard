"""Face database for storing and retrieving face embeddings.

Stores face data in a directory structure:
    face_db/
    ├── metadata.json       # {faces: [{id, name, image_count, enrolled_at}, ...]}
    ├── embeddings.npy      # Numpy array of all face encodings
    └── images/             # Original enrollment images
        ├── kevin_001.jpg
        └── alice_001.jpg
"""

import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    """Record for an enrolled face."""

    id: str  # Unique identifier (sanitized name)
    name: str  # Display name
    image_count: int  # Number of enrolled images
    enrolled_at: str  # ISO timestamp
    embedding_indices: list[int]  # Indices into the embeddings array


class FaceDatabase:
    """Database for storing and managing face embeddings."""

    def __init__(self, db_path: Path):
        """Initialize the face database.

        Args:
            db_path: Path to the face database directory.
        """
        self.db_path = db_path
        self.metadata_path = db_path / "metadata.json"
        self.embeddings_path = db_path / "embeddings.npy"
        self.images_path = db_path / "images"

        # In-memory state
        self._faces: dict[str, FaceRecord] = {}
        self._embeddings: NDArray[np.floating] | None = None  # float32 or float64
        self._embedding_to_face: dict[int, str] = {}  # embedding index -> face id

        # Ensure directories exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load database from disk."""
        # Load metadata
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for face_data in data.get("faces", []):
                    record = FaceRecord(**face_data)
                    self._faces[record.id] = record
                    for idx in record.embedding_indices:
                        self._embedding_to_face[idx] = record.id

                logger.info("Loaded %d faces from database", len(self._faces))
            except Exception as e:
                logger.error("Failed to load face metadata: %s", e)
                self._faces = {}

        # Load embeddings
        if self.embeddings_path.exists():
            try:
                self._embeddings = np.load(self.embeddings_path)
                logger.info("Loaded %d embeddings from database", len(self._embeddings))
            except Exception as e:
                logger.error("Failed to load embeddings: %s", e)
                self._embeddings = None

    def _save(self) -> None:
        """Save database to disk."""
        try:
            # Save metadata
            data = {
                "faces": [asdict(record) for record in self._faces.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Save embeddings
            if self._embeddings is not None and len(self._embeddings) > 0:
                np.save(self.embeddings_path, self._embeddings)
            elif self.embeddings_path.exists():
                self.embeddings_path.unlink()

            logger.debug("Saved face database")
        except Exception as e:
            logger.error("Failed to save face database: %s", e)

    @staticmethod
    def _sanitize_id(name: str) -> str:
        """Convert name to a safe filesystem ID."""
        import re
        s = name.strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_-]", "", s)
        return s or "unknown"

    def add_face(
        self,
        name: str,
        embeddings: list[NDArray[np.floating]],
        images: Optional[list[tuple[str, bytes]]] = None,
    ) -> tuple[bool, str]:
        """Add a new face to the database.

        Args:
            name: Display name for the person.
            embeddings: List of face encoding vectors (128-d each).
            images: Optional list of (filename, image_bytes) to store.

        Returns:
            Tuple of (success, message).
        """
        if not embeddings:
            return False, "No face embeddings provided"

        face_id = self._sanitize_id(name)

        # Check if face already exists
        if face_id in self._faces:
            return False, f"Face '{name}' already exists. Delete it first to re-enroll."

        # Determine embedding indices
        if self._embeddings is None or len(self._embeddings) == 0:
            start_idx = 0
            self._embeddings = np.array(embeddings)
        else:
            start_idx = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, embeddings])

        embedding_indices = list(range(start_idx, start_idx + len(embeddings)))

        # Create face record
        record = FaceRecord(
            id=face_id,
            name=name,
            image_count=len(embeddings),
            enrolled_at=datetime.now().isoformat(),
            embedding_indices=embedding_indices,
        )

        # Update mappings
        self._faces[face_id] = record
        for idx in embedding_indices:
            self._embedding_to_face[idx] = face_id

        # Save images if provided
        if images:
            for i, (filename, img_bytes) in enumerate(images):
                ext = Path(filename).suffix or ".jpg"
                img_path = self.images_path / f"{face_id}_{i:03d}{ext}"
                try:
                    img_path.write_bytes(img_bytes)
                except Exception as e:
                    logger.warning("Failed to save image %s: %s", filename, e)

        # Persist to disk
        self._save()

        logger.info("Enrolled face '%s' with %d embeddings", name, len(embeddings))
        return True, f"Successfully enrolled {name} with {len(embeddings)} images"

    def remove_face(self, name_or_id: str) -> tuple[bool, str]:
        """Remove a face from the database.

        Args:
            name_or_id: Name or ID of the face to remove.

        Returns:
            Tuple of (success, message).
        """
        # Try to find by ID first, then by name
        face_id = self._sanitize_id(name_or_id)
        if face_id not in self._faces:
            # Try exact match on name
            for fid, record in self._faces.items():
                if record.name.lower() == name_or_id.lower():
                    face_id = fid
                    break
            else:
                return False, f"Face '{name_or_id}' not found"

        record = self._faces[face_id]

        # Remove from embedding index mapping
        for idx in record.embedding_indices:
            self._embedding_to_face.pop(idx, None)

        # Remove embeddings (rebuild array without this face's embeddings)
        if self._embeddings is not None:
            # Get indices to keep
            indices_to_remove = set(record.embedding_indices)
            indices_to_keep = [i for i in range(len(self._embeddings)) if i not in indices_to_remove]

            if indices_to_keep:
                new_embeddings = self._embeddings[indices_to_keep]

                # Rebuild index mapping with new indices
                old_to_new = {old: new for new, old in enumerate(indices_to_keep)}
                new_embedding_to_face: dict[int, str] = {}

                for fid, rec in self._faces.items():
                    if fid != face_id:
                        rec.embedding_indices = [old_to_new[i] for i in rec.embedding_indices if i in old_to_new]
                        for idx in rec.embedding_indices:
                            new_embedding_to_face[idx] = fid

                self._embeddings = new_embeddings
                self._embedding_to_face = new_embedding_to_face
            else:
                self._embeddings = None
                self._embedding_to_face = {}

        # Remove face record
        del self._faces[face_id]

        # Remove images
        for img_file in self.images_path.glob(f"{face_id}_*"):
            try:
                img_file.unlink()
            except Exception as e:
                logger.warning("Failed to delete image %s: %s", img_file, e)

        # Persist to disk
        self._save()

        logger.info("Removed face '%s'", record.name)
        return True, f"Successfully removed {record.name}"

    def list_faces(self) -> list[dict]:
        """List all enrolled faces.

        Returns:
            List of face info dictionaries.
        """
        return [
            {
                "id": record.id,
                "name": record.name,
                "image_count": record.image_count,
                "enrolled_at": record.enrolled_at,
            }
            for record in self._faces.values()
        ]

    def get_all_embeddings(self) -> tuple[NDArray[np.floating] | None, list[str]]:
        """Get all embeddings and their corresponding face IDs.

        Returns:
            Tuple of (embeddings array, list of face IDs for each embedding).
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return None, []

        face_ids = [self._embedding_to_face.get(i, "unknown") for i in range(len(self._embeddings))]
        return self._embeddings, face_ids

    def get_face_name(self, face_id: str) -> str:
        """Get display name for a face ID."""
        record = self._faces.get(face_id)
        return record.name if record else face_id

    def get_face_count(self) -> int:
        """Get number of enrolled faces."""
        return len(self._faces)

    def is_empty(self) -> bool:
        """Check if database has no enrolled faces."""
        return len(self._faces) == 0
