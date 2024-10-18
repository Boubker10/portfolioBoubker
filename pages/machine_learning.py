import streamlit as st

def app():
    st.title("Projets Deep Learning")
    st.subheader("Projet 2 : Classification d'images")
    st.write("""
        Classification d'images avec un CNN (réseau de neurones convolutifs).
    """)
    #st.image("assets/projet2_image.png", caption="Exemple de classification d'image")

if __name__ == "__main__":
    app()
