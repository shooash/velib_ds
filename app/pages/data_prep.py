import streamlit as st
import pandas as pd
import plotly.express as px
from velibdslib import get_border, points_to_geo_json, draw_stations_choroplethmap_scatter

def show(): 
    st.title("Exploration des données Vélib' Paris")

    st.markdown("""
    ## 🔍 Contexte du projet
    Ce projet a pour objectif d'optimiser la disponibilité des vélos du service Vélib’ à Paris, en cherchant à prédire les flux de vélos (nombre d’emprunts et de retours) à chaque station.
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
    - Création de la variable `datehour`
    """)

    st.markdown("""
    ### 📉 Valeurs aberrantes : distribution des stations par heure
    """)

    def show_date_hour_station_counts():
        datehour_df = pd.read_hdf('app/data/datehour_stations.h5')
        return px.line(datehour_df, 'datehour', 'station', labels={'datehour' : 'Date-heure', 'station' : '# de stations'}, title="Nombre de stations connues par heure (données d'origine)")

    st.plotly_chart(show_date_hour_station_counts())

    st.markdown("""
    ### 📦 Distribution des valeurs (stations par heure)
    """)

    def show_date_hour_station_boxplot():
        datehour_df = pd.read_hdf('app/data/datehour_stations.h5')
        return px.box(datehour_df, 'station', labels={'station' : 'Nb de station par heure'}, title='Distribution de nombre de stations par heure')

    st.plotly_chart(show_date_hour_station_boxplot())

    st.markdown("""
    ## 📍 Clusterisation spatiale des stations
    Les stations ont été regroupées selon leur proximité géographique à l’aide de **KMeans**.

    Cela permet :
    - de lisser les variations locales,
    - de corriger l’asymétrie du dataset,
    - et d’analyser les zones plutôt que chaque station individuellement.

    Nombre optimal de clusters : 53 à 183 selon la granularité.
    """)

    def show_clusters_map():
        stations = pd.read_hdf('app/data/clusters.h5')
        borders = []
        for l in sorted(stations.labels.unique()):
            borders.append(get_border(stations[stations.labels==l], l))
        geo = points_to_geo_json(borders)
        return draw_stations_choroplethmap_scatter(geo, stations, ret=True)

    st.plotly_chart(show_clusters_map())

    st.markdown("""
    ## 📊 Tendances d’utilisation du service
    Quelques tendances notables observées :
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

    st.markdown("""
    > 🔍 **Limite :** une étude sur l’ensemble de l’année permettrait d’affiner ces observations et de révéler des effets saisonniers plus marqués.
    """)
