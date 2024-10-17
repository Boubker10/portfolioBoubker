import streamlit as st
import cv2
import imutils
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Caffe model
prototxt_path = "pro.txt"  # Replace with your path
model_path = "SSD.caffemodel"  # Replace with your path
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# List of class labels MobileNet SSD was trained to detect
labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
          "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
          "train", "tvmonitor", "cell phone"]

# Generate random colors for each label
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# Define EAR calculation function
def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Streamlit Interface
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["À propos de moi", "Projets Machine Learning", "Projets Deep Learning", "Forecasting Time Series"])

if page == "À propos de moi":
    st.title("À propos de moi")
    st.write("""
        Bonjour, je suis **Boubker Bennani**, passionné par la Data Science et l'Intelligence Artificielle.
        J'ai travaillé sur plusieurs projets liés au Machine Learning, Deep Learning et à la prévision de séries temporelles.
        Mon objectif est de résoudre des problèmes complexes à l'aide des données et d'explorer les dernières avancées dans ces domaines.
        Voici une collection de mes travaux, incluant des modèles d'apprentissage supervisé, des réseaux de neurones profonds et des approches prédictives.
    """)
    st.write("Vous pouvez consulter mon **CV** ici : [Télécharger mon CV](assets/mon_cv.pdf)")

elif page == "Projets Deep Learning":
    st.subheader("Project : SafeDriveVision")
    st.write("""
    **Project Description**: SafeDriveVision is a computer vision project aimed at enhancing road safety. 
    This project leverages deep learning models to detect and alert drivers in real-time about dangerous behaviors, 
    such as using a phone while driving or showing signs of drowsiness. The primary goal is to reduce road accidents 
    by warning drivers of their potentially hazardous actions.
    
    **Features**:
    - **Phone Use Detection**: Utilizes the **MobileNetSSD Caffe** model to identify drivers using their phones while driving.
    - **Drowsiness Detection**: Incorporates a custom detector to monitor signs of driver fatigue.
    - **Real-Time Alerts**: Implements an alert system that warns the driver when risky behavior is detected.
    """)

    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    if st.button('Activate Camera'):
        st.session_state['run_camera'] = True

    if st.button('Deactivate Camera'):
        st.session_state['run_camera'] = False

    if st.session_state['run_camera']:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        stframe = st.empty()

        while cap.isOpened() and st.session_state['run_camera']:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Object detection using Caffe
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            detected_objects = []

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:  # Confidence threshold for detection
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                    detected_objects.append(label)
                    if idx == labels.index("cell phone"):
                        detected_objects.append("Cell Phone Detected")

            # Display detected objects
            info_display = np.zeros((300, 600, 3), dtype=np.uint8)
            for idx, text in enumerate(detected_objects):
                cv2.putText(info_display, text, (10, (idx + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Hand detection processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result_hands = hands.process(frame_rgb)
            result_face_mesh = face_mesh.process(frame_rgb)

            # Face detection processing
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]

            # Detect smiles
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                cv2.putText(frame, "Smile Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Process face mesh
            if result_face_mesh.multi_face_landmarks:
                for face_landmarks in result_face_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
                    
                    landmarks = face_landmarks.landmark
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
                    left_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in left_eye_landmarks]
                    right_eye_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in right_eye_landmarks]

                    left_ear = calculate_ear(left_eye_points)
                    right_ear = calculate_ear(right_eye_points)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < 0.19:
                        cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detected_objects.append("Eyes Closed")
                    else:
                        cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        detected_objects.append("Eyes Open")

            # Process hand detections
            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update the frame on the Streamlit app
            stframe.image(frame, channels="BGR")
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

elif page == "Forecasting Time Series":
    st.title("Forecasting Time Series")
    st.subheader("Projet 3 : Analyse des séries temporelles")
    st.write("Ce projet consiste à prédire des valeurs futures en se basant sur des données historiques.")
