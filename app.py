import streamlit as st
from pages import a_propos, deep_learning, forecasting, machine_learning

# Fonction pour la page d'accueil
def accueil():
    st.title("Bienvenue dans mon portfolio")
    st.write("""
        Bienvenue dans mon portfolio de Data Science et Intelligence Artificielle.
        Ici, vous trouverez une collection de mes travaux sur le Machine Learning, Deep Learning, 
        la prévision de séries temporelles, et bien plus encore.
        
        Explorez les différentes sections pour découvrir mes projets et réalisations.
        Merci de votre visite et bonne exploration !
    """)

# Dictionnaire pour mapper les noms des pages à leurs fonctions correspondantes

if __name__ == "__main__":
    accueil()   
