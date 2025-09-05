from time import sleep
import pandas as pd
import streamlit as st
import requests
import os

DATAPY_API_URL = os.getenv('DATAPY_API_URL', 'http://localhost:8000')
BIG_API_URL = os.getenv('BIG_API_URL', 'http://localhost:8001')

def show():
    st.title("Maintenance")
    st.markdown("Cette page est dédiée à la maintenance du système.")
    if st.button("Voir les logs"):
        with st.expander("Logs du système (dernières 500 lignes)", expanded=True):
            try:
                with open('local/logs/velibds.log', "rb") as log_file:
                    lines = log_file.readlines()[-500:]
                    logs = b"".join(lines).decode("utf-8", errors='replace')
                    # logs = log_file.read().decode("utf-8", errors='replace')
                st.text_area("Logs", logs, height=300, label_visibility="hidden")
                # Add JavaScript to scroll the text area to the end
                st.components.v1.html(
                    """<script>
                    const textArea = parent.document.querySelector('textarea[aria-label="Logs"]');
                    if (textArea) {
                        textArea.scrollTop = textArea.scrollHeight;
                    }
                    </script>""",
                    height=1
                )
            except FileNotFoundError:
                st.write("Fichier de logs non disponible pour le moment.")

    if st.button("Charger les données sur les stations"):
        result = st.empty()
        with result:
            status = st.status("Chargement des données...")
            try:
                response = requests.post(f"{DATAPY_API_URL}/get_stations")
                response.raise_for_status()
                data = response.json()
                status.update(label="Données chargées avec succès.", state='complete')
                with st.expander("Stations", expanded=True):
                    st.dataframe(pd.DataFrame(data.get("stations", [{'station': ''}])).sort_values('station'), hide_index=True)
            except requests.RequestException as e:
                status.update(label=f"Erreur lors du chargement des données : {e}", state='error')
            except Exception as e:
                status.update(label=f"Erreur inattendue : {e}", state='error')
    if st.button("Mettre à jours les données"):
        result = st.empty()
        response = None
        with result:
            try:
                with st.spinner("Mise à jour de données...", show_time=True):
                    response = requests.post(f"{BIG_API_URL}/admin/refresh")
                response.raise_for_status()
                data = response.json()
                with st.expander("Données mises à jour", expanded=True):
                    st.json(data)
            except requests.RequestException as e:
                with st.container():
                    st.error(f"Erreur lors de la mise à jour des données : {e}")
                    if response is not None:
                        st.json(response.json())
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
    if st.button("Réentraîner le modèle"):
        result = st.empty()
        response = None
        with result:
            try:
                with st.spinner("Réentraînement du modèle en cours...", show_time=True):
                    response = requests.post(f"{BIG_API_URL}/admin/retrain")
                response.raise_for_status()
                data = response.json()
                with st.expander("Modèle réentraîné", expanded=True):
                    st.json(data)
            except requests.RequestException as e:
                with st.container():
                    st.error(f"Erreur lors du réentraînement du modèle : {e}")
                    if response is not None:
                        st.json(response.json())
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
