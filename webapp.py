import streamlit as st
from app.pages import intro, data_viz, modeles
from velib_ds.app.pages import data_prep_old

st.set_page_config(page_title="Projet Vélib'", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Introduction", "Data Vizualisation", "Préparation des données", "Modélisation"])

if page == "Introduction":
    intro.show()
elif page == "Data Vizualisation":
    data_viz.show()
elif page == "Préparation des données":
    data_prep_old.show()
elif page == "Modélisation":
    modeles.show()
