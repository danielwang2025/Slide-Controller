""" Slide Controller with Hand Gestures and Voice Commands
This script uses MediaPipe for hand gesture recognition and 
Whisper for voice command recognition to control slide navigation
in presentations. It allows users to go to the next or previous 
slide using hand gestures or voice commands.

How to use:
1. Ensure you have a webcam connected.
2. Install the required libraries:
   pip install opencv-python mediapipe pyautogui sounddevice numpy faster-whisper
3. Run the script and use hand gestures (move right hand to right -> next slide
    or move left hand to left -> previous slide)
    or voice commands like "next slide" or "go back" to navigate slides.
4. Press 'q' to exit the application.
"""

import threading
import time
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import sounddevice as sd
from faster_whisper import WhisperModel

# Initialize Whisper model
model = WhisperModel("small", compute_type="int8")

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)  # pylint: disable=no-member
if not cap.isOpened():
    print("❌ Failed to open webcam")
    sys.exit()

# Slide navigation functions
def go_to_next_slide():
    """ Move to the next slide 
    by simulating a right arrow key press
    """
    print("➡️ Next slide")
    pyautogui.press('right')

def go_to_previous_slide():
    """ Move to the previous slide
    by simulating a left arrow key press
    """
    print("⬅️ Previous slide")
    pyautogui.press('left')

# Audio buffer and configuration
SAMPLE_RATE = 16000
BLOCK_DURATION = 1  # seconds per audio block
BUFFER_SECONDS = 5
Audio_Buffer = deque(maxlen=int(BUFFER_SECONDS / BLOCK_DURATION))  # last 5 seconds of audio

# Audio capture thread
def audio_capture():
    """ Audio capture thread that collects audio data
    and appends it to the audio buffer.
    """
    def callback(indata, status):
        """Audio callback function that processes incoming audio data.

        Args:
            indata: The input audio data.
            status: Status information about the audio stream.
        """
        if status:
            print("⚠️", status)
        Audio_Buffer.append(np.copy(indata[:, 0]))  # use mono channel only

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        callback=callback, blocksize=int(SAMPLE_RATE * BLOCK_DURATION)):
        print("🎙️ Audio stream started...")
        while True:
            sd.sleep(100)

# Audio recognition thread
def audio_recognition():
    """ Audio recognition thread that listens for voice commands
    and triggers slide navigation based on recognized commands.
    """
    print("🤖 Voice recognition started (say 'next slide' or 'go back')")
    while True:
        if len(Audio_Buffer) == 0:
            time.sleep(0.5)
            continue

        audio_data = np.concatenate(list(Audio_Buffer))
        Audio_Buffer.clear()

        # Transcribe using Whisper
        segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
        for segment in segments:
            command = segment.text.strip().lower()
            print(f"🗣️ Recognized command: {command}")
            if "next" in command:
                go_to_next_slide()
            elif "back" in command:
                go_to_previous_slide()

# Start audio threads
threading.Thread(target=audio_capture, daemon=True).start()
threading.Thread(target=audio_recognition, daemon=True).start()

# Gesture control variables
PREV_X = None
LAST_ACTION_TIME = 0
COOLDOWN = 0.5  # seconds

# Main loop (gesture control)
while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from webcam")
        break

    img = cv2.flip(img, 1)  # pylint: disable=no-member
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handType in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            x = handLms.landmark[0].x
            current_time = time.time()
            hand_label = handType.classification[0].label  # "Left" or "Right"

            if PREV_X is not None and (current_time - LAST_ACTION_TIME > COOLDOWN):
                dx = x - PREV_X
                if hand_label == "Right" and dx > 0.1:
                    go_to_next_slide()
                    LAST_ACTION_TIME = current_time
                elif hand_label == "Left" and dx < -0.1:
                    go_to_previous_slide()
                    LAST_ACTION_TIME = current_time

            PREV_X = x

    cv2.imshow("🎬 Slide Controller (Gesture + Voice)", img)  # pylint: disable=no-member

    if cv2.waitKey(1) & 0xFF == ord('q'):  # pylint: disable=no-member
        break

cap.release()
cv2.destroyAllWindows()  # pylint: disable=no-member
