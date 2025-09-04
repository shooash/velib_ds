import streamlit as st

INTRO = """
L'application de prédiction de la demande pour les services Vélib' se base sur 4 service principaux.
- **BIG API**: service principal de l'entrainement de modèles avec des librairies NVidia pour des flux optimisés;
- **DataPy API**: service "léger" pour effectuer les prédictions (peut être répliqué si nécessaire);
- **Monitoring**: groupe de service pour monitorer l'état de l'application, y compris MLFlow, Prometheus et Grafana avec des notifications Telegram;
- **Scheduling**: AirFlow pour planifier le réentrainement de modèles et la mise à jour de datasets historiques.
""".strip()

OUTRO = """
Les données principales pour l'entrainement de modèle sont récupérées via l'API Vélib' dans le dataset historique sur le serveur PostgreSQL.
La base de données et le service de la récupération de données sont hébergés sur Google Cloud Platform.

Les données climatologiques utilisées pour l'entrainement de modèle proviennent de l'API MétéoFrance.

Les prévisions météorologiques nécessaires pour les prédictions sont chargées depuis le service OpenWeatherMap.
""".strip()

def show():
    st.title("Déploiement du service")
    st.markdown(INTRO)
    st.image('app/data/VelibDS_schema.png', caption="L'architecture du service", width="stretch")
    st.markdown(OUTRO)