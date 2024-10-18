import streamlit as st

def show():
    st.title("À propos de moi")
    st.write("""
        Bonjour, je suis **Boubker Bennani**, passionné par la Data Science et l'Intelligence Artificielle.
        J'ai travaillé sur plusieurs projets liés au Machine Learning, Deep Learning et à la prévision de séries temporelles.
        Mon objectif est de résoudre des problèmes complexes à l'aide des données et d'explorer les dernières avancées dans ces domaines.
        Voici une collection de mes travaux.
    """)

    # Lire le fichier PDF
    with open("assets/mon_cv.pdf", "rb") as file:
        cv_pdf = file.read()

    # Bouton de téléchargement
    st.download_button(label="Télécharger mon CV", data=cv_pdf, file_name="Boubker_Bennani_CV.pdf", mime="application/pdf")


if __name__ == "__main__":
    show()
