import streamlit as st

def app():
    st.title("Forecasting Time Series")
    st.subheader("Projet 3 : Prévisions de séries temporelles")
    st.write("""
        Ce projet utilise des méthodes avancées pour prévoir les tendances futures à partir des séries temporelles.
    """)
    st.image("assets/projet_forecasting_image.png", caption="Exemple de prévision")
