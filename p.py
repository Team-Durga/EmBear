import subprocess
import cv2
import numpy as np
import time
from fer import FER
from PIL import Image, ImageDraw, ImageFont
import os

# --- PiTFT beállítás ---
FRAMEBUFFER = "/dev/fb0"
WIDTH = 240
HEIGHT = 135

# --- FER setup ---
detector = FER()

# --- Emoji mapping (egyszínű karakterek) ---
emojis = {
    "angry": "😠",
    "disgust": "😖",
    "fear": "😧",
    "happy": "😃",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}

# --- Betűtípus ---
try:
    font_path= "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_small = ImageFont.truetype(font_path, 48)
    font_big   = ImageFont.truetype(font_path, 48)
except:
    font_small = ImageFont.load_default()
    font_big   = ImageFont.load_default()
# --- RGB565 konverzió ---
def rgb_to_565(img):
    img = img.convert('RGB')
    arr = np.array(img)
    r = (arr[:,:,0] >> 3).astype(np.uint16)
    g = (arr[:,:,1] >> 2).astype(np.uint16)
    b = (arr[:,:,2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.tobytes()

# --- Kép kirajzolása a framebufferre ---
def show_on_fb(emotion, emoji_text):
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Emoji + érzelem
    draw.text((10, 10), emoji_text, font=font_small, fill=(0, 0, 100))
    draw.text((10, 60), emotion, font=font_big, fill=(0, 0, 100))

    # Írás framebufferre RGB565-ben
    with open(FRAMEBUFFER, "wb") as f:
        f.write(rgb_to_565(image))

# --- Libcamera folyamat ---
command = [
    "libcamera-vid",
    "--inline",
    "-t", "0",
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "-o", "-",
    "--codec", "yuv420"
]

proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

print("Indul az érzelem-felismerés. Ctrl+C a kilépéshez.")

try:
    while True:
        raw = proc.stdout.read(640 * 480 * 3 // 2)
        if not raw:
            print("Nem sikerült képet olvasni.")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(480 * 1.5), 640))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        result = detector.top_emotion(frame)
        if isinstance(result, tuple) and len(result) == 2 and all(result):
            emotion, score = result
            print(f"Észlelt: {emotion} ({score:.2f})")
            emoji = emojis.get(emotion, ":|")
            show_on_fb(emotion, emoji)
        else:
            print("Nem észlelhető arc vagy érzelem.")

        time.sleep(2)

except KeyboardInterrupt:
    print("Leállítva.")

finally:
    proc.terminate()
