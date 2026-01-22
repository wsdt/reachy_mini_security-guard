"""Security configuration management.

Follows the same pattern as the main app's .env-based configuration,
storing security settings in the instance_path/.env file.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for the security system."""

    # Armed state
    armed: bool = False

    # Recognition settings
    recognition_hz: float = 2.0  # How often to run face recognition (1-10 Hz)
    confidence_threshold: float = 0.6  # Match confidence (0.4-0.9, lower = stricter)

    # Cooldown settings (in minutes)
    unknown_cooldown_min: float = 5.0  # Time before re-alerting for unknown faces
    greeting_cooldown_min: float = 10.0  # Time before re-greeting same person

    # Paths (set at runtime)
    face_db_path: Optional[Path] = field(default=None, repr=False)

    def to_env_dict(self) -> dict[str, str]:
        """Convert config to environment variable format."""
        return {
            "SECURITY_ARMED": str(self.armed).lower(),
            "SECURITY_RECOGNITION_HZ": str(self.recognition_hz),
            "SECURITY_CONFIDENCE_THRESHOLD": str(self.confidence_threshold),
            "SECURITY_UNKNOWN_COOLDOWN_MIN": str(self.unknown_cooldown_min),
            "SECURITY_GREETING_COOLDOWN_MIN": str(self.greeting_cooldown_min),
        }

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load config from environment variables."""
        return cls(
            armed=os.getenv("SECURITY_ARMED", "false").lower() == "true",
            recognition_hz=float(os.getenv("SECURITY_RECOGNITION_HZ", "2.0")),
            confidence_threshold=float(os.getenv("SECURITY_CONFIDENCE_THRESHOLD", "0.6")),
            unknown_cooldown_min=float(os.getenv("SECURITY_UNKNOWN_COOLDOWN_MIN", "5.0")),
            greeting_cooldown_min=float(os.getenv("SECURITY_GREETING_COOLDOWN_MIN", "10.0")),
        )


def _get_default_face_db_path() -> Path:
    """Get default face database path when instance_path is not available."""
    home = Path.home()
    default_path = home / ".reachy_mini_security" / "face_db"
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def load_security_config(instance_path: Optional[str] = None) -> SecurityConfig:
    """Load security configuration from instance .env file.

    Args:
        instance_path: Path to the instance directory containing .env file.
                      If None, uses environment variables and default paths.

    Returns:
        SecurityConfig with loaded settings.
    """
    # Try to load from instance .env first
    if instance_path:
        env_path = Path(instance_path) / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=str(env_path), override=True)
                logger.debug("Loaded security config from %s", env_path)
            except Exception as e:
                logger.warning("Failed to load .env from %s: %s", env_path, e)

    # Create config from environment
    config = SecurityConfig.from_env()

    # Set face database path
    if instance_path:
        config.face_db_path = Path(instance_path) / "face_db"
    else:
        config.face_db_path = _get_default_face_db_path()

    # Ensure face_db directory exists
    if config.face_db_path:
        config.face_db_path.mkdir(parents=True, exist_ok=True)

    return config


def save_security_config(config: SecurityConfig, instance_path: Optional[str] = None) -> bool:
    """Save security configuration to instance .env file.

    Args:
        config: SecurityConfig to save.
        instance_path: Path to the instance directory.

    Returns:
        True if saved successfully, False otherwise.
    """
    if not instance_path:
        logger.warning("No instance_path provided, cannot persist security config")
        return False

    try:
        env_path = Path(instance_path) / ".env"

        # Read existing .env content
        existing_lines: list[str] = []
        if env_path.exists():
            existing_lines = env_path.read_text(encoding="utf-8").splitlines()

        # Get new values
        new_values = config.to_env_dict()

        # Update or append each security setting
        for key, value in new_values.items():
            # Update environment variable in current process
            os.environ[key] = value

            # Find and replace in existing lines, or mark for append
            found = False
            for i, line in enumerate(existing_lines):
                if line.strip().startswith(f"{key}="):
                    existing_lines[i] = f"{key}={value}"
                    found = True
                    break

            if not found:
                existing_lines.append(f"{key}={value}")

        # Write back
        final_text = "\n".join(existing_lines) + "\n"
        env_path.write_text(final_text, encoding="utf-8")
        logger.info("Saved security config to %s", env_path)
        return True

    except Exception as e:
        logger.error("Failed to save security config: %s", e)
        return False
