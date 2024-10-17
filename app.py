import streamlit as st
import cv2
import imutils
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
model_points = np.array([
    (0.0, 0.0, 0.0),  # Bout du nez
    (-30.0, -125.0, -30.0),  # Coin gauche de l'œil
    (30.0, -125.0, -30.0),  # Coin droit de l'œil
    (-60.0, -70.0, -60.0),  # Coin gauche de la bouche
    (60.0, -70.0, -60.0),  # Coin droit de la bouche
    (0.0, -330.0, -65.0)    # Menton
])

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  
    ear = (A + B) / (2.0 * C)
    return ear


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
Mouth = [61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "cell phone", "cow", 
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
FACE_68_LANDMARKS = [  1, 33, 61, 291, 199, 263, 362, 385, 387, 373, 380, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61 ]


# Barre de navigation pour choisir la page
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["À propos de moi", "Projets Machine Learning", "Projets Deep Learning", "Forecasting Time Series"])

# Charger le contenu de la page sélectionnée
if page == "À propos de moi":
    st.title("À propos de moi")
    st.write("""
        Bonjour, je suis **Boubker Bennani**, passionné par la Data Science et l'Intelligence Artificielle.
        J'ai travaillé sur plusieurs projets liés au Machine Learning, Deep Learning et à la prévision de séries temporelles.
        Mon objectif est de résoudre des problèmes complexes à l'aide des données et d'explorer les dernières avancées dans ces domaines.
        Voici une collection de mes travaux, incluant des modèles d'apprentissage supervisé, des réseaux de neurones profonds et des approches prédictives.
    """)

    st.write("""
        Vous pouvez consulter mon **CV** ici : 
        [Télécharger mon CV](assets/mon_cv.pdf)
    """)


elif page == "Projets Machine Learning":
    st.title("Projets Machine Learning")
    st.subheader("Projet 1 : Prédiction des ventes")
    st.write("""
        Ce projet consiste à prédire les ventes d'un magasin en utilisant des algorithmes de Machine Learning.
        Le modèle analyse les tendances historiques des ventes et prévoit les performances futures.
    """)
    # Visualisation interactive pour le projet 1
    data = {
        "Mois": ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin"],
        "Ventes Prévues": [1000, 1500, 1300, 1700, 1600, 1800]
    }
    import plotly.express as px
    fig1 = px.bar(x=data["Mois"], y=data["Ventes Prévues"], labels={'x':'Mois', 'y':'Ventes Prévues'})
    st.plotly_chart(fig1)

elif page == "Projets Deep Learning":
    st.subheader("Project 4: SafeDriveVision")
    st.write("""
    **Project Description**: SafeDriveVision is a computer vision project aimed at enhancing road safety. 
    This project leverages deep learning models to detect and alert drivers in real-time about dangerous behaviors, 
    such as using a phone while driving or showing signs of drowsiness. The primary goal is to reduce road accidents 
    by warning drivers of their potentially hazardous actions.

    **Features**:
    - **Phone Use Detection**: Utilizes the **YOLOv5** model to identify drivers using their phones while driving.
    - **Drowsiness Detection**: Incorporates a custom detector (sharp detector) to monitor signs of driver fatigue.
    - **Real-Time Alerts**: Implements an alert system that warns the driver when risky behavior is detected.

    For more details on this project, check out the Medium article: [SafeDriveVision: Enhancing Road Safety through Computer Vision](https://medium.com).

    **See the project on GitHub**: [SafeDriveVision on GitHub](https://github.com/Boubker10/SafeDriveVision)
    """)

    # Initialize session state for camera control
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    # Button to toggle the camera
    if st.button('Activate Camera'):
        st.session_state['run_camera'] = True

    if st.button('Deactivate Camera'):
        st.session_state['run_camera'] = False

    if st.session_state['run_camera']:
        # Open the camera
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        stframe = st.empty()  # Placeholder for the video frames
        
        while cap.isOpened() and st.session_state['run_camera']:
            ret, frame = cap.read()
            detected_objects = []
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Resize the frame for processing
            frame = imutils.resize(frame, width=600)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Hand detection processing
            result_hands = hands.process(frame_rgb)
            result_face_mesh = face_mesh.process(frame_rgb)
            #face detection processing
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 3)
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
        
        # Detect smiles
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                result_face_mesh = face_mesh.process(frame_rgb)
            
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
                    
                    if ear < 0.25:
                        detected_objects.append("Eyes Closed")
                    else:
                        detected_objects.append("Eyes Open")

            # Process hand detections
            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update the frame on the Streamlit app
            stframe.image(frame, channels="BGR")

            # Allow a small break to ensure Streamlit updates the UI
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

elif page == "Forecasting Time Series":
    st.title("Forecasting Time Series")
    st.subheader("Projet 3 : Analyse des séries temporelles")
    st.write("""
        Ce projet consiste à prédire des valeurs futures en se basant sur des données historiques.
        J'utilise des techniques avancées de modélisation pour améliorer la précision des prévisions.
    """)
    # st.image("assets/projet_forecasting_image.png", caption="Exemple de prévision de séries temporelles")
