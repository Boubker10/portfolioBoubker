import streamlit as st

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
# Project 4: SafeDriveVision
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

    #st.image("assets/projet2_image.png", caption="Exemple de classification d'image")

elif page == "Forecasting Time Series":
    st.title("Forecasting Time Series")
    st.subheader("Projet 3 : Analyse des séries temporelles")
    st.write("""
        Ce projet consiste à prédire des valeurs futures en se basant sur des données historiques.
        J'utilise des techniques avancées de modélisation pour améliorer la précision des prévisions.
    """)
    #st.image("assets/projet_forecasting_image.png", caption="Exemple de prévision de séries temporelles")
