import streamlit as st
import plotly.express as px
import pandas as pd

def show_date_hour_station_counts():
    datehour_df = pd.read_hdf('app/data/datehour_stations.h5')
    return px.line(datehour_df, 'datehour', 'station', labels={'datehour' : 'Date-heure', 'station' : '# de stations'}, title="Nombre de stations connues par heure (données d'origine)")

def show():
    st.plotly_chart(show_date_hour_station_counts())