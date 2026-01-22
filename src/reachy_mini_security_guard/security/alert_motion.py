"""Alert motion for security guard behavior.

A custom Move that performs an alert/alarm sequence:
1. Quick head shake (left-right-left)
2. Antennas go up in alert position
3. Look around suspiciously
4. Return to vigilant neutral
"""

import logging

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose


logger = logging.getLogger(__name__)


class AlertMove(Move):  # type: ignore
    """Alert motion sequence for intruder detection."""

    def __init__(self, intensity: float = 1.0):
        """Initialize the alert move.

        Args:
            intensity: Motion intensity multiplier (0.5-1.5).
        """
        self.intensity = max(0.5, min(1.5, intensity))

        # Motion parameters (scaled by intensity)
        self.shake_amplitude = np.deg2rad(25) * self.intensity  # Head shake angle
        self.antenna_alert_angle = np.deg2rad(45) * self.intensity  # Antennas up
        self.look_around_yaw = np.deg2rad(30) * self.intensity  # Look left/right

        # Timing (total ~4 seconds)
        self.shake_duration = 0.8  # Quick shake phase
        self.alert_hold = 0.5  # Hold alert pose
        self.look_left_duration = 0.6  # Look left
        self.look_right_duration = 1.0  # Look right (through center)
        self.return_duration = 0.8  # Return to vigilant
        self.vigilant_hold = 0.3  # Hold vigilant pose

        self._duration = (
            self.shake_duration +
            self.alert_hold +
            self.look_left_duration +
            self.look_right_duration +
            self.return_duration +
            self.vigilant_hold
        )

    @property
    def duration(self) -> float:
        """Duration property required by Move interface."""
        return self._duration

    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function (ease in-out cubic)."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def _shake_wave(self, t: float, frequency: float = 3.0) -> float:
        """Generate a shake wave that starts and ends at 0."""
        # Damped sine wave
        envelope = np.sin(np.pi * t)  # Envelope: 0 -> 1 -> 0
        wave = np.sin(2 * np.pi * frequency * t)
        return envelope * wave

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate the alert move at time t.

        Returns:
            Tuple of (head_pose_4x4, antennas_array, body_yaw).
        """
        # Phase boundaries
        t1 = self.shake_duration
        t2 = t1 + self.alert_hold
        t3 = t2 + self.look_left_duration
        t4 = t3 + self.look_right_duration
        t5 = t4 + self.return_duration

        # Default values
        yaw = 0.0
        pitch = 0.0
        antenna_angle = 0.0
        body_yaw = 0.0

        if t < t1:
            # Phase 1: Quick head shake
            phase_t = t / t1
            yaw = self.shake_amplitude * self._shake_wave(phase_t, frequency=4.0)
            # Antennas start going up
            antenna_angle = self.antenna_alert_angle * self._ease_in_out(phase_t)

        elif t < t2:
            # Phase 2: Hold alert pose
            antenna_angle = self.antenna_alert_angle
            # Slight upward tilt (vigilant)
            pitch = np.deg2rad(-5) * self.intensity

        elif t < t3:
            # Phase 3: Look left
            phase_t = (t - t2) / self.look_left_duration
            eased_t = self._ease_in_out(phase_t)
            yaw = self.look_around_yaw * eased_t
            antenna_angle = self.antenna_alert_angle
            pitch = np.deg2rad(-5) * self.intensity

        elif t < t4:
            # Phase 4: Look right (sweep through center)
            phase_t = (t - t3) / self.look_right_duration
            eased_t = self._ease_in_out(phase_t)
            # Go from left (+yaw) to right (-yaw)
            yaw = self.look_around_yaw * (1 - 2 * eased_t)
            antenna_angle = self.antenna_alert_angle
            pitch = np.deg2rad(-5) * self.intensity

        elif t < t5:
            # Phase 5: Return to vigilant neutral
            phase_t = (t - t4) / self.return_duration
            eased_t = self._ease_in_out(phase_t)
            # From right position back to center
            yaw = -self.look_around_yaw * (1 - eased_t)
            # Antennas return to slight alert
            antenna_angle = self.antenna_alert_angle * (1 - 0.5 * eased_t)
            pitch = np.deg2rad(-5) * self.intensity * (1 - eased_t)

        else:
            # Phase 6: Hold vigilant pose
            antenna_angle = self.antenna_alert_angle * 0.5  # Half-alert
            yaw = 0.0
            pitch = 0.0

        # Create head pose
        head_pose = create_head_pose(
            x=0,
            y=0,
            z=0,
            roll=0,
            pitch=pitch,
            yaw=yaw,
            degrees=False,
            mm=False,
        )

        # Antennas (opposite directions for alert look)
        antennas = np.array([antenna_angle, -antenna_angle], dtype=np.float64)

        return (head_pose, antennas, body_yaw)


class VigilantBreathingMove(Move):  # type: ignore
    """A vigilant breathing move for when security is armed but idle.

    Similar to normal breathing but with slightly raised antennas
    and occasional subtle glances.
    """

    def __init__(self):
        """Initialize vigilant breathing."""
        # Breathing parameters
        self.breathing_z_amplitude = 0.003  # Subtle breathing
        self.breathing_frequency = 0.15  # Slightly faster than normal

        # Antenna parameters
        self.antenna_base = np.deg2rad(15)  # Slightly raised
        self.antenna_sway_amplitude = np.deg2rad(8)
        self.antenna_frequency = 0.3

        # Occasional glance parameters
        self.glance_amplitude = np.deg2rad(10)
        self.glance_period = 8.0  # Glance every ~8 seconds

    @property
    def duration(self) -> float:
        """Infinite duration (continuous)."""
        return float("inf")

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate vigilant breathing at time t."""
        # Breathing motion
        z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * t)

        # Occasional subtle glance
        glance_phase = (t % self.glance_period) / self.glance_period
        if 0.4 < glance_phase < 0.6:
            # Quick glance left then right
            local_t = (glance_phase - 0.4) / 0.2
            yaw = self.glance_amplitude * np.sin(2 * np.pi * local_t)
        else:
            yaw = 0.0

        head_pose = create_head_pose(
            x=0, y=0, z=z_offset,
            roll=0, pitch=0, yaw=yaw,
            degrees=False, mm=False,
        )

        # Antennas: base position + subtle sway
        sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * t)
        antennas = np.array([
            self.antenna_base + sway,
            self.antenna_base - sway,
        ], dtype=np.float64)

        return (head_pose, antennas, 0.0)
