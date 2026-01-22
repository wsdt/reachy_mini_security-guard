"""Entrypoint for the Reachy Mini Security Guard app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_security_guard.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini Security Guard app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini Security Guard app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_security_guard.moves import MovementManager
    from reachy_mini_security_guard.console import LocalStream
    from reachy_mini_security_guard.openai_realtime import OpenaiRealtimeHandler
    from reachy_mini_security_guard.tools.core_tools import ToolDependencies
    from reachy_mini_security_guard.audio.head_wobbler import HeadWobbler

    # Security imports
    from reachy_mini_security_guard.security.config import (
        SecurityConfig,
        load_security_config,
        save_security_config,
    )
    from reachy_mini_security_guard.security.security_monitor import SecurityMonitor
    from reachy_mini_security_guard.security.event_handler import SecurityEventHandler
    from reachy_mini_security_guard.security.routes import mount_security_routes

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Security Guard App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Check if running in simulation mode without --gradio
    if robot.client.get_status()["simulation_enabled"] and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode."
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    # Initialize security system
    security_config = load_security_config(instance_path)
    security_monitor: Optional[SecurityMonitor] = None
    security_event_handler: Optional[SecurityEventHandler] = None

    if camera_worker is not None:
        try:
            security_event_handler = SecurityEventHandler()
            security_monitor = SecurityMonitor(
                camera_worker=camera_worker,
                config=security_config,
                on_event=security_event_handler.on_security_event,
            )
            logger.info("Security monitor initialized with %d enrolled faces", security_monitor.database.get_face_count())
        except ImportError as e:
            logger.warning("Face recognition not available: %s", e)
            logger.warning("Install with: pip install face_recognition")
            security_monitor = None
            security_event_handler = None
        except Exception as e:
            logger.error("Failed to initialize security monitor: %s", e)
            security_monitor = None
            security_event_handler = None
    else:
        logger.warning("Camera not available, security monitoring disabled")

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(
        deps,
        gradio_mode=args.gradio,
        instance_path=instance_path,
        security_event_handler=security_event_handler,
    )

    stream_manager: gr.Blocks | LocalStream | None = None

    # Mount security routes to settings app
    if settings_app is not None:
        def get_security_monitor() -> Optional[SecurityMonitor]:
            return security_monitor

        def get_instance_path() -> Optional[str]:
            return instance_path

        def get_security_config() -> SecurityConfig:
            return security_config

        def set_security_config(config: SecurityConfig) -> None:
            nonlocal security_config
            security_config = config
            if security_monitor is not None:
                security_monitor.update_config(config)

        mount_security_routes(
            settings_app,
            get_security_monitor,
            get_instance_path,
            get_security_config,
            set_security_config,
        )

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Reachy Mini Security Guard"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()
    if security_monitor:
        security_monitor.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        # Stop security monitor first
        if security_monitor:
            security_monitor.stop()

        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniSecurityGuardApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the Security Guard app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini Security Guard app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniSecurityGuardApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
