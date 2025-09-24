 -*- coding: utf-8 -*-
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2
import time
import numpy as np

def rgb888_to_rgb565(img):
    """PIL.Image (RGB) -> bytes (RGB565)"""
    arr = np.array(img, dtype=np.uint8)
    r = (arr[...,0] >> 3).astype(np.uint16)
    g = (arr[...,1] >> 2).astype(np.uint16)
    b = (arr[...,2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565.astype('<u2').tobytes() # little-endian sorrend

FRAMEBUFFER = "/dev/fb0"

#emotion detection model
emotion_path = "model.tflite"
emotion_model = Interpreter(model_path=emotion_path)
emotion_model.allocate_tensors()
emotion_input = emotion_model.get_input_details()
emotion_output = emotion_model.get_output_details()
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#camera
picam2 = Picamera2()
picam2.start()
time.sleep(2)

#font type
# font = ImageFont.load_default()
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_size = 22
font = ImageFont.truetype(font_path, font_size)

#main loop
while True:
    frame = picam2.capture_array()  # NumPy array, BGR
    h, w, _ = frame.shape

    face_img = Image.fromarray(frame).convert('L')  #gray
    face_resized = face_img.resize((224,224))
    face_array = np.expand_dims(np.array(face_resized).astype('float32')/255.0, axis=0)
    face_array = np.repeat(face_array[..., np.newaxis], 3, axis=-1)

    # TFLite predication
    emotion_model.set_tensor(emotion_input[0]['index'], face_array)
    emotion_model.invoke()
    output_data = emotion_model.get_tensor(emotion_output[0]['index'])
    emotion_label = emotions[np.argmax(output_data[0])]

    #display pic
    img_display = Image.new('RGB', (240,135), color=(0,0,0))
    draw = ImageDraw.Draw(img_display)
    draw.text((10,60), f"Emotion: {emotion_label}", fill=(0, 200, 255), font=font)

    #shown on the framebuffer
    img_resized = img_display.resize((240,135)).convert('RGB')
    img_rgb565 = rgb888_to_rgb565(img_resized)

    with open(FRAMEBUFFER, "wb") as fb:
        fb.write(img_rgb565)

    time.sleep(0.1)