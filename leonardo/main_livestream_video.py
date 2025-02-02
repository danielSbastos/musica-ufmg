# ajustar para ser trigger com distancia
# crop nas maos antes de identificar
# media de classificacao de X frames 
import random
import mediapipe as mp
import time #import time, sleep, time_ns
import cv2
from pyo import *
import mediapipe as mp
from playsound import playsound
import tensorflow as tf
from pythonosc import udp_client
import random

sender = udp_client.SimpleUDPClient('127.0.0.1', port=9000)

def play_section_1():
    print("SECTION 1")
    sender.send_message('/var', 'audios/ambiente-criancas1.wav')

def play_section_2():
    print("SECTION 2")
    sender.send_message('/var', "audios/soco1.wav")

def play_section_3():
    print("SECTION 3")
    sender.send_message('/var', 'audios/ambiente-criancas2.wav')

def play_section_4():
    print("SECTION 4")
    sender.send_message('/var', "audios/soco2.wav")

def play_random():
    rand = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sender.send_message('/var', f"audios/{rand}.wav")

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

global inside_min_distance
inside_min_distance = False

global begun_playing
begun_playing = None

global last_trigger
last_trigger = time.time()

global section
section = 0

def get_distance(hand_landmarks):
    distance_pinky = transform_coordinates(hand_landmarks, 20)
    distance_wrist = transform_coordinates(hand_landmarks, 0)
    distance_thumb = transform_coordinates(hand_landmarks, 4)
    return min([distance_pinky, distance_wrist, distance_thumb])

def print_result(result):
    global section
    global begun_playing
    global last_trigger 
    global inside_min_distance

    hand_landmarks = result.hand_landmarks

    delta = time.time() - (begun_playing or time.time())
    print(delta)
    print(section)
    if (section == 1) and (delta > 100):
        section = 2
        begun_playing = time.time()
        play_section_3()

    if (section == 3) and (delta > 100):
        section = 4
        begun_playing = time.time()
        # end
   
    if result.gestures and len(result.gestures) >= 2:
        if (section == 0) and begun_playing is None:
            begun_playing = time.time()
            play_section_1()
        
        delta = time.time() - (begun_playing or time.time())
        if (section == 0) and (delta > 60):
            section = 1
            begun_playing = time.time()
            play_section_2()
               
        if (section == 2) and (delta > 60):
            section = 3
            begun_playing = time.time()
            play_section_4()
             
        distance = get_distance(hand_landmarks)
        left_and_right = list(map(lambda x: x[0].category_name, result.handedness))
        left_and_right.sort()
        best_gesture = result.gestures
        best_gesture.sort(key=lambda x: x[0].score, reverse=True)

        current_inside_min_distance = best_gesture[0][0].score > 0.60 and distance < 0.3 and left_and_right == ['Left', 'Right'] 

        if (not inside_min_distance) and current_inside_min_distance:
            gesture = best_gesture[0][0].category_name
            if gesture in ['soco', 'bate']:
                print(f"GESTURE: {gesture}")
                if (section in [0, 2]) and (time.time() - last_trigger) > 0.5:
                    last_trigger = time.time()
                    play_random()

            inside_min_distance = True
            
        if not current_inside_min_distance:
            inside_min_distance = False
    
    print("=========================")


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

    dists = []
    if len(coords) == 2:
        dists.append(distance(coords, 0, 1))
    elif len(coords) == 3:
        dists.append(distance(coords, 0, 1))
        dists.append(distance(coords, 0, 2))
        dists.append(distance(coords, 2, 1))

    if len(dists) == 0:
        return 1
    return min(dists) 

cap = cv2.VideoCapture(0)

global mirrored_frame
last_gesture = ''
ct_frames = 0
ct_results = []
while cap.isOpened():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='./exported_model/gesture_recognizer.task'),
        num_hands=4,
        min_hand_detection_confidence=0.6,
        running_mode=VisionRunningMode.VIDEO)
    with GestureRecognizer.create_from_options(options) as recognizer:
        ret, frame = cap.read()
        if not ret:
            continue
        
        mirrored_frame = cv2.flip(frame, 1)
        #combined_frame = cv2.hconcat([mirrored_frame, frame])
        #image_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = recognizer.recognize_for_video(mp_image, mp.Timestamp.from_seconds(time.time()).value)
        print_result(result)

        cv2.imshow('Hand Gesture Recognition',mirrored_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()