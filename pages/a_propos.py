import streamlit as st

def app():
    st.title("À propos de moi")
    st.write("""
        Je suis un passionné de Data Science et d'IA. Ce site présente mes projets personnels dans le domaine.
        Vous trouverez des projets en Machine Learning, Deep Learning, et Time Series Forecasting.
    """)
    st.image("assets/mon_image.png", caption="Photo personnelle")
