import streamlit as st
import plotly.express as px
import pandas as pd
from velibdslib import get_border, points_to_geo_json, draw_stations_choroplethmap_scatter


def show_date_hour_station_counts():
    datehour_df = pd.read_hdf('app/data/datehour_stations.h5')
    return px.line(datehour_df, 'datehour', 'station', labels={'datehour' : 'Date-heure', 'station' : '# de stations'}, title="Nombre de stations connues par heure (données d'origine)")

def show_date_hour_station_boxplot():
    datehour_df = pd.read_hdf('app/data/datehour_stations.h5')
    return px.box(datehour_df, 'station', labels={'station' : 'Nb de station par heure'}, title='Distribution de nombre de stations par heure')

def show_clusters_map():
    stations = pd.read_hdf('app/data/clusters.h5')
    borders = []
    for l in sorted(stations.labels.unique()):
        borders.append(get_border(stations[stations.labels==l], l))
    geo = points_to_geo_json(borders)
    return draw_stations_choroplethmap_scatter(geo, stations, ret=True)
    
def show():
    
    st.plotly_chart(show_date_hour_station_counts())
    
    st.plotly_chart(show_date_hour_station_boxplot())
    
    st.plotly_chart(show_clusters_map())
    