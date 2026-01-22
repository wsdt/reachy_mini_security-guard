"""Tool for triggering intruder alert motion.

Used by the AI when unknown faces are detected to perform
the alert motion sequence.
"""

import logging
from typing import Any, Dict

from reachy_mini_security_guard.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class AlertIntruder(Tool):
    """Trigger an alert motion sequence for intruder detection."""

    name = "alert_intruder"
    description = (
        "Trigger an alert motion when an unknown person is detected. "
        "Performs a head shake, raises antennas, and looks around suspiciously. "
        "Use this when reporting unknown faces."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "intensity": {
                "type": "number",
                "description": "Motion intensity from 0.5 (subtle) to 1.5 (dramatic). Default is 1.0.",
                "minimum": 0.5,
                "maximum": 1.5,
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute the alert motion.

        Args:
            deps: Tool dependencies including movement_manager.
            **kwargs: Optional 'intensity' parameter.

        Returns:
            Status dictionary.
        """
        intensity = kwargs.get("intensity", 1.0)

        logger.info("Tool call: alert_intruder (intensity=%.1f)", intensity)

        try:
            # Import here to avoid circular imports
            from reachy_mini_security_guard.security.alert_motion import AlertMove

            # Create and queue the alert move
            alert_move = AlertMove(intensity=intensity)
            deps.movement_manager.queue_move(alert_move)

            # Mark as moving for the duration
            deps.movement_manager.set_moving_state(alert_move.duration)

            return {
                "status": "alert_triggered",
                "duration": f"{alert_move.duration:.1f}s",
                "message": "Alert motion queued",
            }

        except Exception as e:
            logger.exception("Failed to trigger alert motion")
            return {
                "status": "error",
                "error": str(e),
            }
