# Slide Controller with Hand Gestures and Voice Commands

This project allows you to control slide navigation in presentations using hand gestures and voice commands. It leverages **MediaPipe** for hand gesture recognition and **Whisper** for voice command recognition.

## Features

- **Hand Gesture Control**:
  - Move your right hand to the right to go to the next slide.
  - Move your left hand to the left to go to the previous slide.
- **Voice Command Control**:
  - Say "next slide" to move to the next slide.
  - Say "go back" to move to the previous slide.
- **Real-time Processing**:
  - Uses your webcam for gesture recognition.
  - Captures audio in real-time for voice command recognition.

## Requirements

Ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries (see `requirements.txt`):
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pyautogui`
  - `sounddevice`
  - `faster-whisper`

## Installation

1. Clone this repository or download the script.
2. Install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your webcam is connected and functional.

## Usage
1. Run the script:

    ```bash
    python main.py
    ```

2. Use the following controls:
    - **Hand Gestures**:
        - Move your right hand to the right → Next slide.
        - Move your left hand to the left → Previous slide.
    - **Voice Commands**:
        - Say "next slide" → Next slide.
        - Say "go back" → Previous slide.
3. Press `q` to exit the application.

## How It Works
- **Hand Gesture Recognition**:
    - Uses MediaPipe's hand tracking solution to detect hand landmarks.
    - Tracks the movement of the hand to determine gestures.
- **Voice Command Recognition**:
    - Captures audio in real-time using sounddevice.
    - Processes audio with the Whisper model to recognize commands.

## Troubleshooting
- **Webcam Issues**:
    - Ensure your webcam is connected and not being used by another application.
    - If the script fails to open the webcam, check your system's camera permissions.
- **Audio Issues**:
    - Ensure your microphone is connected and functional.
    - Check your system's microphone permissions if audio capture fails.

## License
This project is licensed under the MIT License. See the LICENSE file for details.