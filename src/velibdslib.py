import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.express as px
from plotly.subplots import make_subplots

def draw_fig(data, task, title = None, legend = None, xaxis = None, yaxis = None, ret = False):
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
    sec_y_active = any([t.get('secondary_y') for t in task])
    fig = make_subplots(specs=[[{"secondary_y": sec_y_active}]])

    for t in task:
        x = t['x']
        y = t['y']
        fill = t.get('fill')
        color = t.get('color', 'black')
        name = t.get('name')
        secondary_y = t.get('secondary_y', False)
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[y],
                fill='tonexty' if fill else None,
                fillcolor=fill,
                mode='lines+markers',
                line={'color': color},
                name=name
            ), secondary_y=secondary_y)
    fig.update_layout(
            title = title,
            legend_title_text = legend,
            xaxis = xaxis,
            yaxis = yaxis or {}
    )
    if ret:
        return fig
    fig.show()

def draw_stations_choroplethmap_scatter(geojson : dict, data : pd.DataFrame, labels : str = 'labels', center : dict = {'lat': 48.85, 'lon': 2.35}, ret = False, title = "Clusters de stations Vélib'"):
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(geojson=geojson, locations=data[labels], z=data[labels], marker_opacity=0.5))
    fig.add_trace(
        go.Scattermap(
            lon=data['lon'],
            lat=data['lat'],
            mode='markers',
            marker=dict(size=10, color=data[labels]),
            hovertext=data[labels],
            customdata=data[[labels, 'station']].to_numpy(),
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
        title = title
        )

    if not ret:
        fig.show()
    else:
        return fig

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

