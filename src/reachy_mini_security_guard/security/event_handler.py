"""Security event handler for integrating with OpenAI realtime.

Handles security events (greetings, alarms) and injects them
into the conversation context.
"""

import logging
import asyncio
from typing import Any, Optional, Callable
from queue import Queue, Empty

from reachy_mini_security_guard.security.security_monitor import SecurityEvent


logger = logging.getLogger(__name__)


class SecurityEventHandler:
    """Handles security events and bridges them to the conversation system."""

    def __init__(self):
        """Initialize the event handler."""
        self._event_queue: "Queue[SecurityEvent]" = Queue()
        self._connection: Any = None  # OpenAI realtime connection
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_connection(self, connection: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Set the OpenAI realtime connection for sending messages.

        Args:
            connection: The OpenAI realtime connection object.
            loop: The asyncio event loop to use for async operations.
        """
        self._connection = connection
        self._loop = loop

    def on_security_event(self, event: SecurityEvent) -> None:
        """Callback for security monitor events.

        This is called from the security monitor thread, so we queue
        the event for processing in the async context.

        Args:
            event: The security event to handle.
        """
        self._event_queue.put(event)
        logger.debug("Queued security event: %s", event.event_type)

        # Try to process immediately if we have a connection
        if self._connection is not None and self._loop is not None:
            try:
                # Schedule the async processing in the event loop
                asyncio.run_coroutine_threadsafe(
                    self._process_event(event),
                    self._loop,
                )
            except Exception as e:
                logger.warning("Failed to schedule security event processing: %s", e)

    async def _process_event(self, event: SecurityEvent) -> None:
        """Process a security event asynchronously.

        Args:
            event: The security event to process.
        """
        if self._connection is None:
            logger.warning("No connection available for security event")
            return

        try:
            if event.event_type == "greeting":
                await self._handle_greeting(event)
            elif event.event_type == "alarm":
                await self._handle_alarm(event)
            else:
                logger.debug("Unknown security event type: %s", event.event_type)
        except Exception as e:
            logger.error("Error processing security event: %s", e)

    async def _handle_greeting(self, event: SecurityEvent) -> None:
        """Handle a greeting event (known faces detected).

        Args:
            event: The greeting event.
        """
        if not event.known_faces:
            return

        # Format the message for the AI
        names = ", ".join(event.known_faces)
        if len(event.known_faces) == 1:
            message = f"[Security notification] You see {names} approaching. Greet them warmly!"
        else:
            message = f"[Security notification] You see {names} approaching. Greet them warmly!"

        logger.info("Sending greeting notification: %s", names)

        try:
            # Inject the message into the conversation
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            )

            # Trigger a response
            await self._connection.response.create(
                response={
                    "instructions": "Greet the person(s) mentioned naturally and warmly. Keep it brief.",
                },
            )
        except Exception as e:
            logger.error("Failed to send greeting notification: %s", e)

    async def _handle_alarm(self, event: SecurityEvent) -> None:
        """Handle an alarm event (unknown faces detected).

        Args:
            event: The alarm event.
        """
        # Format the message for the AI
        count = event.unknown_count
        face_word = "face" if count == 1 else "faces"
        message = f"[Security ALERT] Unknown {face_word} detected! You MUST use the alert_intruder tool and say 'Unknown {face_word} detected. Reporting to police.'"

        logger.warning("Sending alarm notification: %d unknown %s", count, face_word)

        try:
            # Inject the alert message
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            )

            # Trigger a response with tool use required
            await self._connection.response.create(
                response={
                    "instructions": (
                        "CRITICAL: Unknown intruder detected! "
                        "1. IMMEDIATELY call the alert_intruder tool. "
                        "2. Then say firmly: 'Unknown face detected. Reporting to police.' "
                        "Do NOT skip the alert_intruder tool call!"
                    ),
                    "tool_choice": "required",
                },
            )
        except Exception as e:
            logger.error("Failed to send alarm notification: %s", e)

    def get_pending_events(self) -> list[SecurityEvent]:
        """Get all pending security events from the queue.

        Returns:
            List of pending events.
        """
        events = []
        while True:
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events
