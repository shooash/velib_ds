import streamlit as st
from app.pages import intro, modeles, data_prep, conclusion

st.set_page_config(page_title="Projet Vélib'", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Introduction", "Analyse exploratoire des données", "Modélisation", "Conclusion"])

if page == "Introduction":
    intro.show()
elif page == "Analyse exploratoire des données":
    data_prep.show()
elif page == "Modélisation":
    modeles.show()
elif page == "Conclusion":
    conclusion.show()
