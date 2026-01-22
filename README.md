---
title: Reachy Mini Security Guard
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Security guard with face recognition for Reachy Mini
tags:
 - reachy_mini
 - reachy_mini_python_app
 - security
 - face_recognition
---

# Reachy Mini Security Guard

A security guard application for the Reachy Mini robot with face recognition capabilities. The robot can recognize enrolled faces, greet known visitors by name, and alert when unknown faces are detected.

## Features

- **Face Recognition**: Enroll faces via image upload and recognize them in real-time
- **Personalized Greetings**: Known visitors are greeted by name
- **Intruder Alert**: Unknown faces trigger an alert with motion and verbal warning
- **Dashboard Control**: Web UI for arming/disarming, enrolling faces, and configuring settings
- **Configurable Settings**:
  - Recognition frequency (1-10 Hz)
  - Confidence threshold (30-90%)
  - Unknown face cooldown (1-30 min)
  - Greeting cooldown (1-60 min)

## Installation

> [!IMPORTANT]
> Before using this app, you need to install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).
> 
> The `face_recognition` library requires `dlib` which needs `cmake` to build:
> - **macOS**: `brew install cmake`
> - **Ubuntu**: `apt install cmake libboost-all-dev`

### Using uv

```bash
uv venv --python 3.12.1
source .venv/bin/activate
uv sync
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`
2. Fill in the required values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. OpenAI API key for conversation. |
| `SECURITY_ARMED` | Optional. Start armed (`true`) or disarmed (`false`). Default: `false` |
| `SECURITY_RECOGNITION_HZ` | Optional. Recognition frequency. Default: `2.0` |
| `SECURITY_CONFIDENCE_THRESHOLD` | Optional. Match confidence (0-1). Default: `0.6` |
| `SECURITY_UNKNOWN_COOLDOWN_MIN` | Optional. Minutes before re-alerting. Default: `5.0` |
| `SECURITY_GREETING_COOLDOWN_MIN` | Optional. Minutes before re-greeting. Default: `10.0` |

## Running the App

```bash
reachy-mini-security-guard
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--head-tracker {yolo,mediapipe}` | `None` | Face tracking backend for head following. |
| `--no-camera` | `False` | Disable camera (disables security features). |
| `--gradio` | `False` | Launch Gradio web UI. |
| `--debug` | `False` | Enable verbose logging. |

### Examples

```bash
# Run with MediaPipe face tracking
reachy-mini-security-guard --head-tracker mediapipe

# Run with Gradio interface
reachy-mini-security-guard --gradio
```

## Dashboard

When running via Reachy Mini Apps, access the settings dashboard to:

1. **Arm/Disarm** the security system
2. **Enroll faces** by uploading images
3. **Manage enrolled faces** (view, delete)
4. **Configure settings** (recognition frequency, cooldowns, confidence)

## How It Works

### Face Enrollment

1. Open the dashboard
2. Enter the person's name
3. Upload one or more clear photos of their face
4. Click "Enroll Face"

For best results:
- Use multiple images (3-5) from different angles
- Ensure good lighting
- Face should be clearly visible

### Security Behavior

**When Armed:**
- Continuously monitors camera feed for faces
- **Known face detected**: Greets the person by name (respects greeting cooldown)
- **Unknown face only**: Triggers alert motion and says "Unknown face detected. Reporting to police."
- **Mixed (known + unknown)**: Greets known faces, no alarm (known person vouches for guest)

**When Disarmed:**
- No face recognition or alerts
- Robot behaves as normal conversation assistant

## Security Profile

The app uses a custom "security_guard" personality profile with:
- Professional, vigilant demeanor
- Warm greetings for known visitors
- Firm responses to unknown intruders
- Uses the `alert_intruder` tool for alarm motion

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Monitor                          │
│  (Separate thread, configurable Hz)                         │
├─────────────────────────────────────────────────────────────┤
│  Camera Worker → Face Detection → Face Recognition          │
│                        ↓                                    │
│              ┌────────┴────────┐                           │
│              │                 │                           │
│        KNOWN FACE         UNKNOWN FACE                     │
│              │                 │                           │
│     Greeting Event       Alarm Event                       │
│              │                 │                           │
│              └────────┬────────┘                           │
│                       ↓                                    │
│            Security Event Handler                          │
│                       ↓                                    │
│           OpenAI Realtime Session                          │
│           (Injects context message)                        │
│                       ↓                                    │
│              AI Response + Tool Call                       │
│              (alert_intruder if alarm)                     │
└─────────────────────────────────────────────────────────────┘
```

## LLM Tools

| Tool | Action |
|------|--------|
| `alert_intruder` | Perform alert motion (head shake, antennas up, look around) |
| `move_head` | Move head to look in a direction |
| `play_emotion` | Play a recorded emotion |
| `head_tracking` | Enable/disable face tracking |
| `dance` | Perform a dance |

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run linting
ruff check .

# Run tests
pytest
```

## License

Apache 2.0
