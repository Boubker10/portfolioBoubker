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
    st.title("Projets Deep Learning")
    st.subheader("Projet 2 : Classification d'images")
    st.write("""
        Ce projet utilise un réseau de neurones convolutifs (CNN) pour classifier des images.
        Le modèle est capable de reconnaître des objets tels que des voitures, des animaux, et des personnes dans des images.
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
