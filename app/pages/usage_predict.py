import datetime
import streamlit as st
from src.velibdslib import get_border, points_to_geo_json, draw_stations_choroplethmap_scatter, draw_fig
import pandas as pd
import requests
import os
from streamlit_calendar import calendar

DATAPY_API_URL = os.getenv('DATAPY_API_URL', 'http://localhost:8000')

def show_clusters_map(stations : pd.DataFrame):
    borders = []
    for l in sorted(stations.cluster.unique()):
        borders.append(get_border(stations[stations.cluster==l], l))
    geo = points_to_geo_json(borders)
    return draw_stations_choroplethmap_scatter(geo, stations, ret=True, title="Réseau Velib' de Paris", labels='cluster')

    # if has_on_click is not None:
    #     def handle_click_labels(trace, points, selector):
    #         print(points)
    #         if points.point_inds:
    #             i = points.point_inds[0]
    #             data = trace.locations[i]     # [label]
    #             on_click_labels(data)

    #     def handle_click_stations(trace, points, selector):
    #         print(points)
    #         if points.point_inds:
    #             i = points.point_inds[0]
    #             data = trace.customdata[i]     # [label, station]
    #             on_click_stations(data)
    #     if on_click_labels is not None:
    #         fig.data[0].on_click(handle_click_labels)
    #     if on_click_stations is not None:
    #         fig.data[1].on_click(handle_click_stations)

@st.cache_data
def get_stations():
    try:
        response = requests.post(f"{DATAPY_API_URL}/get_stations")
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data['stations'])
    except requests.RequestException as e:
        st.error(f"Error fetching stations: {e}")
        return pd.DataFrame({'station': [], 'lat': [], 'lon': [], 'name': [], 'cluster': []})

def show_prediction_stats(pred_df : pd.DataFrame):
    '''
    Show the prediction results and related counts
    '''
    pred_df['taken'] = pred_df['prediction'].where(pred_df['prediction'] < 0, 0)
    pred_df['returned'] = pred_df['prediction'].where(pred_df['prediction'] > 0, 0)
    st.markdown('## Résultats de prédiction')
    st.dataframe(pred_df[['name', 'station', 'datehour', 'taken', 'returned', 'prediction']], hide_index=True)
    st.markdown('## En total')
    row = st.container(horizontal=True)
    with row:
        st.metric("Vélos pris", value=abs(round(pred_df['taken'].sum())), border=True)
        st.metric("Vélos retournés", value=round(pred_df['returned'].sum()), border=True)
    flux = round(pred_df['prediction'].sum(), 2)
    st.metric("Flux de vélos", value=f"{'+' if flux >= 0 else '-'}{abs(round(flux))}",
        chart_data=pred_df.groupby(['datehour'])['prediction'].sum(),
        chart_type="area", border=True, delta=flux)
    st.markdown('## Statistique journalière')
    daily_df = pred_df.groupby('datehour')[['prediction', 'taken', 'returned']].sum().reset_index()
    daily_df['datehour'] = pd.to_datetime(daily_df['datehour'])
    daily_df['Date'] = daily_df['datehour'].dt.date
    daily_df['morning'] = daily_df['prediction'].where(daily_df['datehour'].dt.hour < 12, 0)
    daily_df['afternoon'] = daily_df['prediction'].where(daily_df['datehour'].dt.hour >= 12, 0)

    daily_df = daily_df.groupby('Date').agg(
        m=('morning', 'sum'), 
        a=('afternoon', 'sum'), 
        t=('prediction', 'sum'), 
        taken=('taken', 'sum'), 
        returned=('returned', 'sum')).reset_index()

    # Weekdays
    weekdays = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    daily_df['dayofweek'] = daily_df['Date'].apply(lambda x: weekdays[x.weekday()])
    # Order
    daily_df = daily_df[['Date', 'dayofweek', 'taken', 'returned', 't', 'm', 'a']].sort_values('Date')
    # Formats
    daily_df['taken'] = daily_df['taken'].abs().round()
    daily_df['returned'] = daily_df['returned'].round()
    daily_df['m'] = daily_df['m'].round()
    daily_df['a'] = daily_df['a'].round()
    # Readable names
    daily_df = daily_df.rename(columns=
        {
            'm' : 'Flux matin',
            'a' : 'Flux après-midi',
            't' : 'Flux prédit',
            'taken' : 'Vélos pris',
            'returned' : 'Vélos retourné',
            'dayofweek' : 'Jour',
        }
    )
    st.dataframe(daily_df, hide_index=True)
    # Draw chart
    fig = draw_fig(daily_df, [
        {'x' : 'Date', 'y' : 'Flux matin', 'fill' : None, 'color' : 'blue', 'name' : 'Flux matin'},
        {'x' : 'Date', 'y' : 'Flux prédit', 'fill' : 'rgba(0, 80, 100, 0.2)', 'color' : 'gray', 'name' : 'Flux du jour'},
        {'x' : 'Date', 'y' : 'Flux après-midi', 'fill' : 'rgba(100, 0, 80, 0.2)', 'color' : 'red', 'name' : 'Flux après-midi'},
    ],
    title = 'Utilisation de vélos prédite par jour',
    ret = True
    )
    st.plotly_chart(fig)
    
def predict(stations_df : pd.DataFrame):
    '''Run a prediction for the list of stations as argument'''
    calendar_selection = st.session_state.get('calendar_selection')
    if not calendar_selection or 'select' not in calendar_selection:
        st.error("Il faut selectionner les heures ou les dates.")
        return
    from_dt_str = calendar_selection['select']['start']
    to_dt_str = calendar_selection['select']['end']
    from_dt = datetime.datetime.strptime(from_dt_str, "%Y-%m-%dT%H:%M:%S.%fZ") 
    to_dt = datetime.datetime.strptime(to_dt_str, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.timedelta(hours=1)
    datehours = list(pd.date_range(from_dt, to_dt, freq="h").strftime("%Y-%m-%d %H:%M"))
    stations_list = stations_df['station'].unique().tolist()
    status_holder = st.empty()
    status = status_holder.status(f"Prédiction pour {len(stations_list)} station(s) entre {from_dt} et {to_dt} ({len(datehours)} heures).")
    response = None
    try:
        response = requests.post(f"{DATAPY_API_URL}/predict", json={
            "station": stations_list,
            "date": datehours
        })
        response.raise_for_status()
        data = response.json()
        if 'station' not in data:
            raise ValueError("No station in data")
        status.update(label=f"Prédiction faite avec {len(data['station'])} enregistrement(s).", state='complete')
        if not data:
            raise Exception("Failed to predict!")
        pred_df = pd.DataFrame(data)
        # Add names
        pred_df = pred_df.merge(stations_df[['station', 'name']], on='station', how='left')
        show_prediction_stats(pred_df)
        
    except requests.RequestException as e:
        error_description = response.json() if response is not None else 'No response.'
        status.update(label=f"Error fetching stations: {e} : {error_description}", state='error')
        return None

def predict_stations(selected_stations : pd.DataFrame):
    '''
    Run a prediction for selected number of stations
    '''
    stations_df = selected_stations[selected_stations['name'].isin(st.session_state.get('stations_select', []))]
    if not len(stations_df):
        st.error("Il faut selectionner des stations.")
        return
    predict(stations_df)

def predict_clusters(selected_stations : pd.DataFrame):
    '''
    Run a prediction for selected number of clusters
    '''
    stations_df = selected_stations[selected_stations['cluster'].isin(st.session_state.get('clusters_select', []))]
    if not len(stations_df):
        st.error("Il faut selectionner des clusters.")
        return
    predict(stations_df)

def show():
    st.title("Velib' de Paris: Utilisation du Réseau")
    st.markdown("""Cette application permet d'estimer la demande pour le service "Vélib'". Sélectionnez les stations sur la carte pour voir les détails et prédire leur usage.""")
    stations_df = get_stations()
    selection = st.plotly_chart(show_clusters_map(stations_df), on_select='rerun', selection_mode=('points', 'box'))
    if selection and selection.selection.points:
        # Get selected clusters and stations
        clusters = [p['location'] for p in selection.selection.points if 'location' in p]
        clusters += [p['customdata'][0] for p in selection.selection.points if 'customdata' in p]
        clusters = list(sorted(set(clusters)))
        stations = list(sorted(set([p['customdata'][1] for p in selection.selection.points if 'customdata' in p])))
        if st.session_state.get('action_select') == "Prédire les stations":
            selected_stations = stations_df[stations_df['station'].isin(stations)]
        else:
            selected_stations = stations_df[stations_df['cluster'].isin(clusters)]

        st.info(f'Vous avez sélectionné {len(clusters)} clusters et {len(stations)} stations.')
        st.dataframe(selected_stations, hide_index=True)

        cols = st.columns([1,2])
        with cols[0]:
            st.selectbox("Que faire ?", ["Prédire les stations", "Prédire les clusters"], key='action_select')
        with cols[1]:
            if st.session_state.get('action_select') == "Prédire les stations":
                st.multiselect("Sélectionner les stations à prédire", selected_stations['name'], default=selected_stations['name'], key='stations_select')
            else:
                st.multiselect("Sélectionner les clusters à prédire", clusters, default=clusters, key='clusters_select')

        # Docs: https://github.com/im-perativa/streamlit-calendar + https://fullcalendar.io/docs/headerToolbar
        calendar_options = {
            "selectable" : True,
            "locale" : "fr",
            "timeZone" : "Europe/Paris",
            "slotLabelFormat": {
                "hour": '2-digit',
                "minute": '2-digit',
                # "second": '2-digit',
                "hour12": False
            },
            "slotDuration" : "01:00",
            "allDaySlot" : False,
            "headerToolbar": {
                "center": "title",
                "left": "today prev,next",
                "right" : "timeGridWeek,dayGridMonth",
            },
            "buttonText": {
                "today": "Aujourd'hui",
                "prev": "Précédent",
                "next": "Suivant",
                "timeGridWeek": "Heures",
                "dayGridMonth": "Jours",
            },
            "initialView": "timeGridWeek",
            "height" : 400,
        }
        calendar(options=calendar_options, key='calendar_selection')
        if st.button("Lancer la prédiction"):
            if st.session_state.get('action_select') == "Prédire les stations":
                predict_stations(selected_stations)
            else:
                predict_clusters(selected_stations)