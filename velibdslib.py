import plotly.graph_objects as go
from velibconnector import VelibConnector
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.express as px
import datetime

def draw_fig(data, task, title = None, legend = None, xaxis = None, yaxis = None):
    """
    Designer un graphique linéaire

    Args:
        data: DataFrame
        task: liste de paramètres. [{x, y, name(optional), color(optional), fill(optional)}]
        title: str Titre de graphique
        legend: str Titre de légende
        xaxis (dict, optional): plotly update_layout xaxis params comme {title, dtick, range, mirror}.
        yaxis (dict, optional): plotly update_layout yaxis params comme {title, dtick, range, mirror}.
    Example de paramètres:
        [ { 
            'x' : 'weekday', 
            'y' : 'delta_mean', 
            'fill' : 'rgba(0, 80, 100, 0.2)', 
            'name' : 'Moyenne',
            'color' : 'blue' } ]
        Plotly designe les graphiques dans l'ordre inverse, donc mieux de mettre la ligne basse en avant pour que le FILL marche proprement.

    """
    fig = go.Figure()
    for t in task:
        x = t['x']
        y = t['y']
        fill = t.get('fill')
        color = t.get('color', 'black')
        name = t.get('name')
        fig.add_traces([
            go.Scatter(
                x=data[x],
                y=data[y],
                fill='tonexty' if fill else None,
                fillcolor=fill,
                mode='lines+markers',
                line={'color': color},
                name=name
            )
        ])
    fig.update_layout(
            title = title,
            legend_title_text = legend,
            xaxis = xaxis,
            yaxis = yaxis or {}
    )
    fig.show()
    
def draw_stations_choroplethmap_scatter(geojson : dict, data : pd.DataFrame, labels : str = 'labels', center : dict = {'lat': 48.85, 'lon': 2.35}):
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(geojson=geojson, locations=data[labels], z=data[labels], marker_opacity=0.5))
    fig.add_trace(
        go.Scattermap(
            lon=data['lon'],
            lat=data['lat'],
            mode='markers',
            marker=dict(size=10, color=data['labels']),
            hovertext=data.labels,
            customdata=data[['labels', 'station']].to_numpy(),
            hovertemplate="<b>Cluster %{customdata[0]:}</b><br>" +
                    "Station: %{customdata[1]}<br>" +
                    "Latitude: %{lat:.2f}<br>" +
                    "Longitude: %{lon:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        width = 800,
        height=600,
        map = {
            'center' : center, 
            'zoom' : 10},
        title = "Clusters de stations Vélib'"
        )

    fig.show()
    
def points_to_geo_json(data : list):
    result = {
        'type' : 'FeatureCollection',
        'features' : []
    }
    for d in data:
        label = d[0][3]
        feat = {
            'type' : 'Feature',
            'properties' : {
            },
            'id' : int(label),
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [[r[0], r[1]] for r in d] + [[d[0][0], d[0][1]]]
                                 ]
                },
        }
        result['features'].append(feat)
    return result

def get_border(df, label):
    if len(df) < 3:
        return [[df['lon'].iloc[0], df['lat'].iloc[0], df['station'].iloc[0], label]]
    points = df[['lon', 'lat']].to_numpy()
    hull = ConvexHull(points)
    return [[df['lon'].iloc[p], df['lat'].iloc[p], df['station'].iloc[p], label] for p in hull.vertices]

def max_distance_in_cluster(data):
    cluster_points = data[['convlat', 'convlon']]
    if len(cluster_points) > 1:
        dist_matrix = distance_matrix(cluster_points, cluster_points)
        return np.max(dist_matrix)
    else:
        return -1

def detect_outliers(selection: pd.Series, k = 1.5, method='IQR'):
    """
    Returns a blacklist of indexes considered as outliers.
    Args:
        - method = 'IQR' (default) ou 'zscore'
        - k - multiply coef for IQR or zscore limit
    """
    if method == 'IQR':
        q = selection.quantile([0.25, 0.75]).to_list()
        IQR = (q[1] - q[0]) * k
        low_border = q[0] - IQR
        high_border = q[1] + IQR
    elif method=='zscale':
        selection_zscore = selection.copy()
        selection_zscore = zscore(selection)
        low_border = selection[selection_zscore < -k].max()
        high_border = selection[selection_zscore > k].min()
    elif method=='hard':
        low_border = k
        high_border = selection.max()
    else:
        raise SyntaxError('Wrong method.')
    print('Valeur min-max:', selection.min(), '-', selection.max())
    print('Seuils de outliers:', low_border, '-', high_border)
    high_outliers = selection[selection>high_border].index.to_list()
    low_outliers = selection[selection<low_border].index.to_list()
    print('Nombre de valeurs total:', len(selection))
    print(f'Grands outliers: {len(high_outliers)} ou {round(len(high_outliers)/len(selection)*100, 2)}%')
    print(f'Petits outliers: {len(low_outliers)} ou {round(len(low_outliers)/len(selection)*100, 2)}%')
    return high_outliers + low_outliers

def convert_lon(lon, min_lon):
    """
    Get meters distance for lon degrees.
    """
    lon -= min_lon
    return lon * 72987
def convert_lat(lat, min_lat):
    """
    Get meters distance for lat degrees.
    """
    lat -= min_lat
    return lat * 111000

def draw_kmeans_silhouette(stations):
    # Calculer l'indice de silhouette pour différents nombres de clusters
    silhouette_scores = []
    for k in range(50, 251):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(stations[['convlat', 'convlon']])
        silhouette_scores.append(silhouette_score(stations[['convlat', 'convlon']], kmeans.labels_))

    # Tracer la courbe de l'indice de silhouette
    px.line(x=range(50, 251), y=silhouette_scores, labels={'x' : 'Nombre de clusters', 'y' : 'Indice de Silhouette'}, title='Indice Silhouette pour determiner le meilleur nombre de clusters pour KMeans').show()

def get_station_clusters(stations, model):
    stations['lat'] = stations.lat.apply(float)
    stations['lon'] = stations.lon.apply(float)
    min_lon = stations.lon.min()
    min_lat = stations.lat.min()
    stations['convlon'] = stations.lon.apply(convert_lon, args=[min_lon])
    stations['convlat'] = stations.lat.apply(convert_lat, args=[min_lat])
    kmeans = model.fit(stations[['convlat', 'convlon']])
    return kmeans.labels_

class VelibData:
    def __init__(self, from_dt : str = '2024-12-05', to_dt : str = None, debug = True):
        if to_dt == 'auto':
            to_dt = datetime.date.today().strftime(r'%Y-%m-%d')
        cmd = f"""
        SELECT status_id, station, lat, lon,
            delta, bikes, capacity, name,
                dt from velib_all
                where dt > '{from_dt}' {"and dt <'" + to_dt + "'" if to_dt else ""}
        """
        self.velib_data = VelibConnector(cmd).to_pandas()

    def transform_velib(self, debug = False):
        df = self.velib_data.copy()
        if debug:
            print(f'{len(df)} lignes chargées pour la période de {df.dt.min().date()} à {df.dt.max().date()}')
        df['datehour'] = df.dt.dt.floor('h')
        if debug:
            print(f"""Suppression de {df.isna().sum().sum()} valeurs manquantes.""")
        df.dropna()
        full_duplicates_count = df.duplicated(['dt', 'bikes', 'capacity', 'station']).sum()
        if debug:
            print(f"""Suppression de {full_duplicates_count} ({round(full_duplicates_count/len(df)*100, 2)}%) doublons d'origine, c'est-à-dire des lignes identiques par les valeurs clé de données de temps réel Vélib: 'dt', 'bikes', 'capacity' et 'station'.""")            
        df.drop_duplicates(['dt', 'bikes', 'capacity', 'station'], inplace=True)
        calc_duplicated = df.duplicated(['datehour', 'station']).sum()
        if debug:
            print(f"""Il y a {calc_duplicated} ({round(calc_duplicated/len(df)*100, 2)}%) lignes qui réprésentent les mêmes stations pour les même dates et heures. On rejoint ces doublons""")
        df = df.sort_values(['dt', 'status_id']).drop_duplicates(['datehour', 'station'])
        df['delta'] = df.groupby('station')['bikes'].diff().fillna(df.delta)
        # Ajout de clusters
        stations = df[['station', 'lat', 'lon', 'name']].drop_duplicates()
        # n_clusters : 53 le meilleur, meilleurs des plus grands = 81, 101, 123, 182
        stations['labels'] = get_station_clusters(stations, KMeans(n_clusters=182, random_state=0))
        df = pd.merge(df, stations[['station', 'labels']], on='station', how='left')
        # Nettoyage des heures outliers
        datehour_df = df.groupby('datehour').labels.nunique()
        q = datehour_df.quantile([0.25, 0.75]).to_list()
        # outliers comme q1 - delta(q3,q1)*1.5 et q3 + delta(q3,q1)*1.5
        q_margin = (q[1] - q[0]) * 1.5
        seuil_bas = q[0] - q_margin
        seuil_haut = q[1] + q_margin
        if debug:
            print('Seuils de outliers:', seuil_bas, '-', seuil_haut)
            print("Nombre de clusters:", df.labels.nunique())
            print('Max clusters par heure:', datehour_df.max())
            print('Min clusters par heure:', datehour_df.min())
        top_outliers = datehour_df[datehour_df>seuil_haut]
        top_outliers_count = top_outliers.count()
        bottom_outliers = datehour_df[datehour_df<seuil_bas]
        bottom_outliers_count = bottom_outliers.count()
        total_hours = df.datehour.nunique()
        if debug:
            print("Nombre d'heures:", total_hours)
            print('Nombre de top outliers:', top_outliers_count, f"{round(top_outliers_count/total_hours*100, 2)}%")
            print('Nombre de bottom outliers:', bottom_outliers_count, f"{round(bottom_outliers_count/total_hours*100, 2)}%")
        blacklist = bottom_outliers.index.to_list() + top_outliers.index.to_list()
        total_rows = len(df)
        dropped_rows = len(df[df.datehour.isin(blacklist)])
        if debug:
            print(f'A enlever {len(blacklist)} outliers, {dropped_rows} lignes soit {round(dropped_rows/total_rows*100, 2)}% de {total_rows} lignes en totale.')
        # manual_cut = seuil_bas - 1
        # blacklist = datehour_df[datehour_df<manual_cut].index.to_list()
        # dropped_rows = len(clean_df[clean_df.datehour.isin(blacklist)])
        # print(f'Si on coupe manuellement les heures avec moins de {manual_cut} clusters on supprime {len(blacklist)} outliers et {dropped_rows} lignes soit {round(dropped_rows/total_rows*100, 2)}% de {total_rows} ligne en totale.')

        df = df[~df.datehour.isin(blacklist)]
        self.velib_data = df
        return self