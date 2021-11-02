import sys
sys.path.append('/home/bilalharoon/.local/lib/python3.9/site-packages')
import cv2
import mediapipe as mp
import bpy
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
index_finger = bpy.data.objects['Armature'].pose.bones['palm.01.L.001']
index_finger.rotation_mode = 'XYZ'

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image_height, image_width, _ = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
                normalized_landmark = hand_landmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, image_width, image_height)
                if point == mp_hands.HandLandmark.INDEX_FINGER_MCP:
                    v1 = [normalized_landmark.x, normalized_landmark.y]
                    
                if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    v2 = [normalized_landmark.x, normalized_landmark.y]
                
                if v1 is not None and v2 is not None:
                    axis = angle(v1, v2)
                    index_finger.rotate_euler.rotate_axis('Z', axis)
                print(point)
                print(pixelCoordinatesLandmark)
                print(normalized_landmark)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      cv2.destroyAllWindows()
      break
cap.release()
