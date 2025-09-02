import streamlit as st
from app.pages import intro, modeles, data_prep, conclusion, usage_predict, deployment, administration

st.set_page_config(page_title="Projet Vélib'", layout="wide")

st.sidebar.title("Navigation")
# page = st.sidebar.radio("Aller à", ["Introduction", "Analyse exploratoire des données", "Modélisation", "Conclusion"])
pages = {
    "Introduction": intro,
    "Exploration des données": data_prep,
    "Modélisation": modeles,
    "Déploiement": deployment,
    # "Conclusion": conclusion,
    "Application": usage_predict,
    "Maintenance" : administration
}

if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'

with st.sidebar:
    for page_name, page_module in pages.items():
        if page_name == st.session_state.get('page', 'Introduction'):
            page_name = "**" + page_name + "**"
        st.button(page_name, 
                  icon=':material/chevron_right:', 
                  type = 'tertiary', 
                  on_click=lambda page_name=page_name: st.session_state.update({'page': page_name.replace("**", "")}))

pages[st.session_state.page].show()
