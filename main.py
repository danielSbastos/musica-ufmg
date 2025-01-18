from time import time, sleep
import random

import cv2
import xml.etree.ElementTree as ET
from pyo import *
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='./exported_model/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(2)
s = Server().boot()
s.start()

sleep(2)

last_gesture = ''

previous = time()
delta = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    recognition_result = recognizer.recognize(mp_image)

    #hand_landmarks = recognition_result.hand_landmarks
    #results.append((top_gesture, hand_landmarks))

    current = time()
    delta += current - previous
    previous = current

    if delta > 1:
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            current_gesture = top_gesture.category_name

            if current_gesture == 'soco':
                freq = random.uniform(50, 150)
                env_table = LinTable([(0, 0), (100, 1), (4096, 0)])
                env = TrigEnv(Trig(), table=env_table, dur=0.6, mul=2)
                osc = SineLoop(freq=freq, feedback=0.2, mul=env)
                lowpass = ButLP(osc, freq=200)
                stereo = lowpass.mix(2) 
                reverb = Freeverb(stereo, size=0.8, damp=0.5, bal=0.3).out()
            elif current_gesture == 'bate':
                freq = random.uniform(200, 400)
                env_table = LinTable([(0, 0), (50, 1), (500, 0)])
                env = TrigEnv(Trig(), table=env_table, dur=0.6, mul=3)
                osc = SineLoop(freq=freq, feedback=0.2, mul=env)
                crispiness = random.uniform(100, 500)
                lowpass = ButLP(osc, freq=crispiness)
                stereo = lowpass.mix(2)
                stereo.out()
            elif current_gesture == 'vira':
                snare_noise = Noise(mul=1.0)
                env_table = LinTable([(0, 0), (50, 1), (3000, 0)])
                env = TrigEnv(Trig(), table=env_table, dur=1.0, mul=3)
                noise_with_envelope = snare_noise * env
                lfo_freq = random.uniform(0.5, 5.0)
                lfo_depth = random.uniform(200, 600)
                lfo = Sine(freq=lfo_freq, mul=lfo_depth)
                oscillating_filter_freq = 400 + lfo
                lowpass = ButLP(noise_with_envelope, freq=oscillating_filter_freq)
                stereo = lowpass.mix(2)
                stereo.out()

            last_gesture = current_gesture
            print("============================")
            print(current_gesture)
        delta = 0

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
