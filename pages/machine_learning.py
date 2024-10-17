import streamlit as st
import plotly.express as px

def app():
    st.title("Projets Machine Learning")
    st.subheader("Projet 1 : Prédiction des ventes")
    st.write("""
        Prédiction des ventes avec des algorithmes de Machine Learning.
    """)
    # Données factices pour la visualisation
    data = {
        "Mois": ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin"],
        "Ventes Prévues": [1000, 1500, 1300, 1700, 1600, 1800]
    }
    fig = px.bar(x=data["Mois"], y=data["Ventes Prévues"], labels={'x':'Mois', 'y':'Ventes Prévues'})
    st.plotly_chart(fig)
