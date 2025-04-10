import streamlit as st

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():

    st.title("Exploration des données Vélib' Paris")

    st.markdown("""
    ## 🔍 Contexte du projet
    Ce projet a pour objectif d'optimiser la disponibilité des vélos du service Vélib’ à Paris, en prédisant les flux de vélos (nombre d’emprunts et de retours) à chaque station.
    """)

    st.markdown("""
    ## 🧱 Construction de la base de données
    Les données ont été extraites des API Open Data de Vélib’ (station_status et station_information) et collectées toutes les heures via un script sur Google Cloud Run.

    Elles sont stockées dans deux tables principales :
    - **velib_status** : état en temps réel des stations (vélos disponibles, bornes, etc.)
    - **velib_stations** : informations statiques (nom, localisation, capacité)

    Ces deux sources ont été fusionnées dans une vue unifiée : `velib_all`, utilisée pour nos analyses.
    """)

    st.markdown("""
    ## 🧹 Prétraitement des données
    - Création d’une variable `delta` représentant la variation de vélos entre deux mises à jour successives
    - Nettoyage des **doublons** complets et partiels
    - Analyse des **valeurs manquantes** et **outliers**
    - Création de la variable `datehour` (troncature de la date à l'heure près)
    """)

    st.markdown("""
    ### ✂️ Exemple de doublons partiels
    Certains enregistrements sont répétés à la même heure avec des valeurs différentes :
    """)

    st.image("images/doublons_partiels.png", caption="Exemples de doublons partiels détectés")

    st.markdown("""
    ### 📉 Valeurs aberrantes
    Boxplots utilisés pour détecter les outliers sur `station` et `datehour` :
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/boxplot_station.png", caption="Outliers sur le nombre de mises à jour par station")
    with col2:
        st.image("images/boxplot_datehour.png", caption="Outliers sur le nombre de stations mises à jour par heure")

    st.markdown("""
    ## 📍 Clusterisation spatiale des stations
    Les stations ont été regroupées selon leur proximité géographique à l’aide de **KMeans**.

    Cela permet :
    - de lisser les variations locales,
    - de corriger l’asymétrie du dataset,
    - et d’analyser les zones plutôt que chaque station individuellement.

    Nombre optimal de clusters : 53 à 183 selon la granularité.
    """)

    st.image("images/clusters.png", caption="Exemple de visualisation des clusters de stations Vélib'")

    st.markdown("""
    ## 📊 Tendances d’utilisation du service
    Voici quelques insights obtenus après nettoyage :
    """)

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/flux_journalier.png", caption="Variation moyenne des flux dans une journée")
    with col4:
        st.image("images/flux_hebdo.png", caption="Flux moyen de vélos par jour de la semaine")

    st.markdown("""
    - Pic d’utilisation entre 5h-8h et 15h-20h
    - Activité réduite en week-end et pendant les fêtes
    - Stations vides plus fréquentes en matinée
    """)

    st.markdown("""
    ## ☁️ Analyse de la corrélation avec la météo
    Nous avons croisé les données Vélib’ avec les données Météo France :
    - Température moyenne horaire (°C)
    - Précipitations horaires (mm)

    Résultats :
    - **Température** : -0.0006
    - **Précipitations** : -0.0024

    Aucune corrélation significative n’a été observée sur la période hivernale.
    """)

    col5, col6 = st.columns(2)
    with col5:
        st.image("images/temp_scatter.png", caption="Impact de la température sur le flux de vélos")
    with col6:
        st.image("images/pluie_boxplot.png", caption="Impact des précipitations sur le flux")

    st.markdown("""
    > 🔍 **Limite :** une étude sur l’ensemble de l’année permettrait d’affiner ces observations et de révéler des effets saisonniers plus marqués.
    """)
