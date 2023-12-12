import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

import requests

import threading
import queue
import time

# Local server URL where Unity will listen for the text
server_url = 'http://localhost:5000/receive_text'

#variables for threading
var = False

q = queue.Queue()
prin = queue.Queue()

#code from keypoint example code
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def f(): #posting to server
    lastSentEmotion = 'Default'
    while var:
        if q.empty():
            time.sleep(0.001)
        else:
            if q.qsize() > 1:
                x = q.get_nowait() #burn items off the queue until 1 left to reduce lag
            m = q.get_nowait()
            if m is not None and m != lastSentEmotion:
                requests.post(server_url, json=m)
                lastSentEmotion=m

def t(): #depredicated funciton
    while var:
        if prin.empty():
            time.sleep(0.001)
        else:
            if prin.qsize()>1:
                print(prin.get_nowait())

cap_device = 0
cap_width = 1920
cap_height = 1080

use_brect = True

#variables to start threading
t1 = threading.Thread(target=f)
t2 = threading.Thread(target=t)

print("Starting thread for server push")
var = True
t1.start()
t2.start()

# Camera preparation
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

# Model load
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9)

keypoint_classifier = KeyPointClassifier()


# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0
lastEmotion = 'Default' #variable to store last emotion detected
#loop to run the code
while True:

    # Process Key (ESC: end)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    # Camera capture
    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            #emotion classification
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            #prin.put_nowait(facial_emotion_id.getResult())
            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(
                    debug_image,
                    brect,
                    keypoint_classifier_labels[facial_emotion_id])
            # make the payload for the server and send it
            if lastEmotion != keypoint_classifier_labels[facial_emotion_id]:
                payload = {'text': keypoint_classifier_labels[facial_emotion_id]}
                q.put_nowait(payload)
                # response = requests.post(server_url, json=payload)
                lastEmotion = payload['text']

    # Screen reflection
    cv.imshow('Facial Emotion Recognition', debug_image)
var = False #tells thread to stop working
t1.join() #cleans up the mess
t2.join()
cap.release()
cv.destroyAllWindows()
#close the windows in open cv when program ends