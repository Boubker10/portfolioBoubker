import streamlit as st
import cv2 
import cv2
import imutils
import numpy as np
import mediapipe as mp
import pytesseract
from PIL import Image
import torch
from scipy.spatial import distance as dist
import math
import easyocr
import pyautogui
import webbrowser

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

prototxt_path = "pro.txt"  
model_path = "SSD.caffemode"  
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
          "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
          "train", "tvmonitor", "cell phone"]

colors = np.random.uniform(0, 255, size=(len(labels), 3))

def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def safedrivevision():
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
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing = mp.solutions.drawing_utils
        
        stframe = st.empty()

        while cap.isOpened() and st.session_state['run_camera']:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            detected_objects = []

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5: 
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


            info_display = np.zeros((300, 600, 3), dtype=np.uint8)
            for idx, text in enumerate(detected_objects):
                cv2.putText(info_display, text, (10, (idx + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            frame = imutils.resize(frame, width=600)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result_hands = hands.process(frame_rgb)
            result_face_mesh = face_mesh.process(frame_rgb)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 3)
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
    
            smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                cv2.putText(frame, "Smile Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
                    
                    if ear < 0.19:
                        cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detected_objects.append("Eyes Closed")
                    else:
                        cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        detected_objects.append("Eyes Open")

            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            stframe.image(frame, channels="BGR")

            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

def snn_reconnaissance_facial():
    st.subheader("Project : Reconnaissance Faciale avec Siamese Neural Networks (SNN)")
    st.write("""
    **Project Description**: This project represents an evolution of a pre-existing facial expression recognition (FER) system by leveraging Spiking Neural Networks (SNNs) and event cameras. We contributed to this field by optimizing the existing source code, training the improved model with the CKPLUS database, and conducting tests and evaluations to measure its performance

    **Repository GitHub**: Retrouvez les détails et le code source du projet sur [GitHub](https://github.com/Boubker10/DeepLearningReconFacialSNN).
    """)



def read_and_annotate(image, lang='fr'):
    # Initialiser le reader EasyOCR avec la langue choisie
    reader = easyocr.Reader([lang])

    # Convertir l'image en tableau NumPy
    img = np.array(image.convert('RGB'))
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convertir en niveaux de gris pour améliorer la reconnaissance
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Utiliser EasyOCR pour détecter et lire le texte dans l'image
    results = reader.readtext(gray)

    # Annoter l'image avec les textes reconnus et retourner uniquement ceux avec une confiance > 0.5
    filtered_texts = []
    for result in results:
        # Vérifier si nous avons bien 3 éléments (bbox, texte, probabilité)
        if len(result) == 3:
            bbox, text, prob = result

            if prob > 0.1:  # Filtrer les résultats avec une confiance > 0.5
                # Extraire les coordonnées de la boîte englobante
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

                # Dessiner un rectangle autour du texte détecté
                cv2.rectangle(img_cv2, top_left, bottom_right, (0, 255, 0), 2)

                # Annoter l'image avec le texte reconnu
                cv2.putText(img_cv2, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Ajouter le texte filtré à la liste des textes reconnus
                filtered_texts.append((text, prob))

    # Convertir l'image annotée de BGR à RGB pour l'affichage
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    return img_pil, filtered_texts

def plat_recon():
    st.title("Détection des Plaques d'Immatriculation avec EasyOCR")

    language = st.selectbox("Sélectionnez la langue pour la reconnaissance de texte", ["français", "anglais", "arabe", "chinois"])
    lang_map = {
        "français": "fr",
        "anglais": "en",
        "arabe": "ar",
        "chinois": "ch_sim"
    }

    selected_lang = lang_map[language] 
    uploaded_file = st.file_uploader("Téléchargez une image pour la détection de plaques", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image d'entrée", use_column_width=True)

        if st.button("Lancer la détection de plaques"):
            result_image, plate_results = read_and_annotate(image, lang=selected_lang)
            st.image(result_image, caption="Résultats de la détection de plaques", use_column_width=True)

            st.subheader(f"Texte extrait des plaques en {language}:")
            if plate_results:
                for text, prob in plate_results:
                    st.write(f"Texte : {text} (Confiance : {prob:.2f})")
            else:
                st.error("Aucune plaque d'immatriculation détectée.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Fonction pour contrôler le Dino avec la main
def control_dino(hand_landmarks, frame):
    for hand_landmark in hand_landmarks:
        for id, lm in enumerate(hand_landmark.landmark):
            if id == 8:  # Index Finger Tip
                if lm.y < 0.5:  # Si la main est en haut de l'écran, faire sauter le Dino
                    pyautogui.press('space')
                    return
    
    # Détection de la peau
    skin_mask = detect_skin(frame)
    blurred = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
    
        count_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(frame, far, 1, [0, 0, 255], -1)
                    cv2.line(frame, start, end, [0, 255, 0], 2)
            
            if count_defects >= 4:
                pyautogui.press('space')
                cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    except ValueError:
        st.write("Aucun contour trouvé")

# Fonction pour détecter la peau
def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return mask

# Interface Streamlit pour le contrôle du Dino Game
def dino_game():
    st.title("Contrôle du Dino avec la détection des mains")

    # Ajouter un bouton pour ouvrir le jeu Dino dans Chrome
    if st.button("Jouer au jeu Dino dans le navigateur"):
        webbrowser.open("https://chromedino.com/")  # Ouvre un site où le jeu Dino est accessible

    # Activer la caméra pour le contrôle du Dino
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    if st.button('Activer la caméra'):
        st.session_state['run_camera'] = True

    if st.button('Désactiver la caméra'):
        st.session_state['run_camera'] = False

    if st.session_state['run_camera']:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened() and st.session_state['run_camera']:
            ret, frame = cap.read()
            if not ret:
                st.write("Erreur: Impossible de lire l'image de la caméra")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                control_dino(results.multi_hand_landmarks, frame)

            stframe.image(image, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

def app():
    st.title("Projets Deep Learning")

    # Liste des projets sous "Deep Learning"
    projet_selectionne = st.selectbox("Sélectionnez un projet", ["SafeDriveVision", "Reconnaissance Faciale (SNN)","licence plate recognition","Dino Game"])

    # Afficher le projet sélectionné
    if projet_selectionne == "SafeDriveVision":
        safedrivevision()
    elif projet_selectionne == "Reconnaissance Faciale (SNN)":
        snn_reconnaissance_facial()
    elif projet_selectionne == "licence plate recognition":
        plat_recon()
    elif projet_selectionne == "Dino Game":
        dino_game()
if __name__ == "__main__":
    app()
