"""FastAPI routes for security settings dashboard.

Provides endpoints for:
- Security status and arm/disarm toggle
- Settings management (recognition frequency, cooldowns, etc.)
- Face enrollment and management
"""

import logging
from typing import Optional, Callable, Any
from pathlib import Path

from reachy_mini_security_guard.security.config import SecurityConfig, save_security_config


logger = logging.getLogger(__name__)


def mount_security_routes(
    app: Any,  # FastAPI app
    get_security_monitor: Callable[[], Any],  # Returns SecurityMonitor or None
    get_instance_path: Callable[[], Optional[str]],
    get_security_config: Callable[[], SecurityConfig],
    set_security_config: Callable[[SecurityConfig], None],
) -> None:
    """Mount security-related routes to the FastAPI app.

    Args:
        app: FastAPI application instance.
        get_security_monitor: Callable that returns the SecurityMonitor instance.
        get_instance_path: Callable that returns the instance path.
        get_security_config: Callable that returns current SecurityConfig.
        set_security_config: Callable to update SecurityConfig.
    """
    try:
        from fastapi import File, Form, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not available, security routes not mounted")
        return

    class SecuritySettingsPayload(BaseModel):
        """Payload for updating security settings."""
        armed: Optional[bool] = None
        recognition_hz: Optional[float] = None
        confidence_threshold: Optional[float] = None
        unknown_cooldown_min: Optional[float] = None
        greeting_cooldown_min: Optional[float] = None

    # ---------- Status & Control ----------

    @app.get("/security/status")
    def get_security_status() -> JSONResponse:
        """Get current security status."""
        monitor = get_security_monitor()
        config = get_security_config()

        if monitor is None:
            return JSONResponse({
                "armed": config.armed,
                "running": False,
                "face_count": 0,
                "recognition_hz": config.recognition_hz,
                "confidence_threshold": config.confidence_threshold,
                "unknown_cooldown_min": config.unknown_cooldown_min,
                "greeting_cooldown_min": config.greeting_cooldown_min,
                "error": "Security monitor not initialized",
            })

        status = monitor.get_status()
        return JSONResponse(status)

    @app.post("/security/arm")
    def arm_security() -> JSONResponse:
        """Arm the security system."""
        monitor = get_security_monitor()
        config = get_security_config()

        config.armed = True
        if monitor is not None:
            monitor.armed = True

        # Persist to .env
        instance_path = get_instance_path()
        if instance_path:
            save_security_config(config, instance_path)

        set_security_config(config)

        return JSONResponse({
            "armed": True,
            "status": "Security system armed",
        })

    @app.post("/security/disarm")
    def disarm_security() -> JSONResponse:
        """Disarm the security system."""
        monitor = get_security_monitor()
        config = get_security_config()

        config.armed = False
        if monitor is not None:
            monitor.armed = False

        # Persist to .env
        instance_path = get_instance_path()
        if instance_path:
            save_security_config(config, instance_path)

        set_security_config(config)

        return JSONResponse({
            "armed": False,
            "status": "Security system disarmed",
        })

    # ---------- Settings ----------

    @app.get("/security/settings")
    def get_security_settings() -> JSONResponse:
        """Get current security settings."""
        config = get_security_config()
        return JSONResponse({
            "armed": config.armed,
            "recognition_hz": config.recognition_hz,
            "confidence_threshold": config.confidence_threshold,
            "unknown_cooldown_min": config.unknown_cooldown_min,
            "greeting_cooldown_min": config.greeting_cooldown_min,
        })

    @app.post("/security/settings")
    def update_security_settings(payload: SecuritySettingsPayload) -> JSONResponse:
        """Update security settings."""
        monitor = get_security_monitor()
        config = get_security_config()

        # Update config with provided values
        if payload.armed is not None:
            config.armed = payload.armed
        if payload.recognition_hz is not None:
            config.recognition_hz = max(0.5, min(10.0, payload.recognition_hz))
        if payload.confidence_threshold is not None:
            config.confidence_threshold = max(0.3, min(0.95, payload.confidence_threshold))
        if payload.unknown_cooldown_min is not None:
            config.unknown_cooldown_min = max(0.5, min(60.0, payload.unknown_cooldown_min))
        if payload.greeting_cooldown_min is not None:
            config.greeting_cooldown_min = max(0.5, min(120.0, payload.greeting_cooldown_min))

        # Update monitor if available
        if monitor is not None:
            monitor.update_config(config)

        # Persist to .env
        instance_path = get_instance_path()
        if instance_path:
            save_security_config(config, instance_path)

        set_security_config(config)

        return JSONResponse({
            "status": "Settings updated",
            "armed": config.armed,
            "recognition_hz": config.recognition_hz,
            "confidence_threshold": config.confidence_threshold,
            "unknown_cooldown_min": config.unknown_cooldown_min,
            "greeting_cooldown_min": config.greeting_cooldown_min,
        })

    # ---------- Face Management ----------

    @app.get("/security/faces")
    def list_faces() -> JSONResponse:
        """List all enrolled faces."""
        monitor = get_security_monitor()

        if monitor is None:
            return JSONResponse({
                "faces": [],
                "error": "Security monitor not initialized",
            })

        faces = monitor.list_faces()
        return JSONResponse({
            "faces": faces,
            "count": len(faces),
        })

    @app.post("/security/faces")
    async def enroll_face(
        name: str = Form(...),
        images: list[UploadFile] = File(...),
    ) -> JSONResponse:
        """Enroll a new face.

        Accepts multipart form with:
        - name: Display name for the person
        - images: One or more image files (JPG, PNG)
        """
        monitor = get_security_monitor()

        if monitor is None:
            raise HTTPException(status_code=503, detail="Security monitor not initialized")

        if not name or not name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required")

        # Read image data
        image_data: list[tuple[str, bytes]] = []
        for img in images:
            try:
                content = await img.read()
                if content:
                    image_data.append((img.filename or "image.jpg", content))
            except Exception as e:
                logger.warning("Failed to read uploaded image: %s", e)

        if not image_data:
            raise HTTPException(status_code=400, detail="Could not read any images")

        # Enroll the face
        success, message = monitor.enroll_face(name.strip(), image_data)

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return JSONResponse({
            "status": "enrolled",
            "name": name.strip(),
            "message": message,
        })

    @app.delete("/security/faces/{name}")
    def remove_face(name: str) -> JSONResponse:
        """Remove an enrolled face."""
        monitor = get_security_monitor()

        if monitor is None:
            raise HTTPException(status_code=503, detail="Security monitor not initialized")

        success, message = monitor.remove_face(name)

        if not success:
            raise HTTPException(status_code=404, detail=message)

        return JSONResponse({
            "status": "removed",
            "name": name,
            "message": message,
        })

    logger.info("Security routes mounted")
