# -*- coding: utf-8 -*-
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2
import time

def rgb888_to_rgb565(img):
    """PIL.Image (RGB) -> bytes (RGB565)"""
    arr = np.array(img, dtype=np.uint8)
    r = (arr[...,0] >> 3).astype(np.uint16)
    g = (arr[...,1] >> 2).astype(np.uint16)
    b = (arr[...,2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.astype('<u2').tobytes() # little-endian sorrend

FRAMEBUFFER = "/dev/fb0"

# Emotion detection model
emotion_path = "model.tflite"
emotion_model = Interpreter(model_path=emotion_path)
emotion_model.allocate_tensors()
emotion_input = emotion_model.get_input_details()
emotion_output = emotion_model.get_output_details()

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
emojis = {
    'Angry': "😠",
    'Disgust': "😖",
    'Fear': "😧",
    'Happy': "😃",
    'Sad': "😢",
    'Surprise': "😲",
    'Neutral': "😐"
}

# Camera
picam2 = Picamera2()
picam2.start()
time.sleep(2)

# Font
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_small = ImageFont.truetype(font_path, 28)
font_big   = ImageFont.truetype(font_path, 37)

# State variables
last_emotion = None
repeat_count = 0

# Main loop
while True:
    frame = picam2.capture_array()  # NumPy array, BGR
    h, w, _ = frame.shape

    face_img = Image.fromarray(frame).convert('L')  # gray
    face_resized = face_img.resize((224,224))
    face_array = np.expand_dims(np.array(face_resized).astype('float32')/255.0, axis=0)
    face_array = np.repeat(face_array[..., np.newaxis], 3, axis=-1)

    # TFLite prediction
    emotion_model.set_tensor(emotion_input[0]['index'], face_array)
    emotion_model.invoke()
    output_data = emotion_model.get_tensor(emotion_output[0]['index'])
    emotion_label = emotions[np.argmax(output_data[0])]

    # Check if repeated
    if emotion_label == last_emotion:
        repeat_count += 1
    else:
        repeat_count = 1
        last_emotion = emotion_label

    if repeat_count >= 2:
        display_text = f"{emojis[emotion_label]} {emotion_label}"
    else:
        display_text = "None"

    # Display pic
    img_display = Image.new('RGB', (240,135), color=(255,255,255))
    draw = ImageDraw.Draw(img_display)
    draw.text((10, 30), "Emotion:", fill=(0, 0, 100), font=font_small)
    draw.text((10, 60), display_text, fill=(0, 0, 100), font=font_big)
    
    # Shown on the framebuffer
    img_resized = img_display.resize((240,135)).convert('RGB')
    img_rgb565 = rgb888_to_rgb565(img_resized)

    with open(FRAMEBUFFER, "wb") as fb:
        fb.write(img_rgb565)

    time.sleep(1
