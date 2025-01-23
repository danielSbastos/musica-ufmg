# ajustar para ser trigger com distancia
# crop nas maos antes de identificar
# media de classificacao de X frames 

import mediapipe as mp

import time #import time, sleep, time_ns
import random

import cv2
import xml.etree.ElementTree as ET
from pyo import *
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from playsound import playsound
from collections import Counter

import os
import absl.logging
import sys
import tensorflow as tf

from pyo import *

import matplotlib.pyplot as plt
import numpy as np

# Create a red screen
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
sys.stderr = open(os.devnull, 'w')
os.environ["MESA_DEBUG"] = "0"
os.environ["MESA_LOG_LEVEL"] = "fatal"
tf.get_logger().setLevel('INFO')

s = Server()
s.setOutputDevice(0)
s.boot()
s.start()

def play_soco():
    playsound("soco.wav", block=False)

def play_bate():
    playsound("bate.wav", block=False)

def play_vira():
    playsound("vira.wav", block=False)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

global inside_min_distance
inside_min_distance = False

def print_result(result, output_image: mp.Image, timestamp_ms: int):
    hand_landmarks = result.hand_landmarks

    if result.gestures and len(result.gestures)>=2:
        distance_pinky = transform_coordinates(hand_landmarks, 20)
        distance_wrist = transform_coordinates(hand_landmarks, 0)
        distance_thumb = transform_coordinates(hand_landmarks, 4)
        distance = min([distance_pinky, distance_wrist, distance_thumb])

        left_and_right = list(map(lambda x: x[0].category_name, result.handedness))
        left_and_right.sort()
        best_gesture = result.gestures
        best_gesture.sort(key=lambda x: x[0].score, reverse=True)

        current_inside_min_distance = best_gesture[0][0].score > 0.66 and distance < 0.14 and left_and_right == ['Left', 'Right'] 

        global inside_min_distance

        if current_inside_min_distance and inside_min_distance:
            print("INSIDE")
            pass # do nothing
        
        if (not inside_min_distance) and current_inside_min_distance:
            print("ENTERED")
            gesture = best_gesture[0][0].category_name
            if gesture == 'soco':
                play_soco()
            elif gesture == 'vira':
                play_vira()
            elif gesture == 'bate':
                play_bate()
            inside_min_distance = True
        
        if not current_inside_min_distance:
            inside_min_distance = False


def distance(coords, a, b):
    x_delta = coords[a][0] - coords[b][0]
    y_delta = coords[a][1] - coords[b][1]
    z_delta = coords[a][2] - coords[b][2]
    return (x_delta**2 + y_delta**2 + z_delta**2)**0.5

def transform_coordinates(set_gestures_coordinates, i):
    coords = []
    for gestures_coordinates in set_gestures_coordinates:
        thumb_mark = gestures_coordinates[i]
        _x, _y, _z = thumb_mark.x, thumb_mark.y, thumb_mark.z
        coords.append([_x, _y, _z])

    dist0_1 = distance(coords, 0, 1)
    return dist0_1

cap = cv2.VideoCapture(2)

global mirrored_frame
last_gesture = ''
ct_frames = 0
ct_results = []
while cap.isOpened():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='./exported_model/gesture_recognizer.task'),
        num_hands=4,
        min_hand_detection_confidence=0.7,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    with GestureRecognizer.create_from_options(options) as recognizer:
        ret, frame = cap.read()
        if not ret:
            continue
        
        mirrored_frame = cv2.flip(frame, 1)
        combined_frame = cv2.hconcat([frame, mirrored_frame])
        image_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        recognizer.recognize_async(mp_image, mp.Timestamp.from_seconds(time.time()).value)

        cv2.imshow('Hand Gesture Recognition', combined_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()