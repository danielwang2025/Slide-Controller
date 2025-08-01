import cv2
import mediapipe as mp
import time
import pyautogui
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from collections import deque

# 初始化 Whisper 模型
model = WhisperModel("small", compute_type="int8")

# 初始化 MediaPipe 手势识别
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 摄像头初始化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头无法打开")
    exit()

# 翻页函数
def go_to_next_slide():
    print("➡️ 翻到下一页")
    pyautogui.press('right')

def go_to_previous_slide():
    print("⬅️ 回到上一页")
    pyautogui.press('left')

# 音频缓存和控制参数
samplerate = 16000
block_duration = 0.5  # 每次采集0.5秒
buffer_seconds = 5
audio_buffer = deque(maxlen=int(buffer_seconds / block_duration))  # 保存最近5秒音频块

# 音频采集线程
def audio_capture():
    def callback(indata, frames, time_info, status):
        if status:
            print("⚠️", status)
        audio_buffer.append(np.copy(indata[:, 0]))  # 只取1通道

    with sd.InputStream(channels=1, samplerate=samplerate, callback=callback, blocksize=int(samplerate * block_duration)):
        print("🎙️ 开始音频采集...")
        while True:
            sd.sleep(100)

# 音频识别线程
def audio_recognition():
    print("🤖 开始语音识别（说 'next slide' 或 'go back'）")
    while True:
        if len(audio_buffer) == 0:
            time.sleep(0.5)
            continue

        audio_data = np.concatenate(list(audio_buffer))
        audio_buffer.clear()  # 清空缓存，准备下一次

        # 推理
        segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
        for segment in segments:
            command = segment.text.strip().lower()
            print(f"🗣️ 识别结果：{command}")
            if "next slide" in command:
                go_to_next_slide()
            elif "go back" in command:
                go_to_previous_slide()

# 启动语音线程
threading.Thread(target=audio_capture, daemon=True).start()
threading.Thread(target=audio_recognition, daemon=True).start()

# 手势翻页变量
prev_x = None
last_action_time = 0
cooldown = 1  # 秒

# 主循环（手势控制）
while True:
    success, img = cap.read()
    if not success:
        print("❌ 摄像头读取失败")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            x = handLms.landmark[0].x
            current_time = time.time()

            if prev_x is not None and (current_time - last_action_time > cooldown):
                dx = x - prev_x
                if dx > 0.1:
                    go_to_next_slide()
                    last_action_time = current_time
                elif dx < -0.1:
                    go_to_previous_slide()
                    last_action_time = current_time

            prev_x = x

    cv2.imshow("🎬 Slide Controller (手势 + 语音)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
