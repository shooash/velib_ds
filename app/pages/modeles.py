import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

@st.cache_resource
def show_chatelet_alignement_figure():
    df_viz = pd.read_hdf(r"app/data/chatelet_une_semaine.h5")
    fig = go.Figure()
    fig.add_scatter(x=df_viz['hour'], y=df_viz['delta'], name="Info d'origine", mode='markers', marker={'color' : df_viz['color']})
    fig.update_xaxes(dtick=1, minallowed=-0.5, maxallowed=23.5, title={'text' : 'Heure'})
    fig.update_yaxes(range=[-10, 10])
    fig.update_layout(title="Alignement des pics d'utilisation de la station Chatelet pour la 7ème semaine 2025")
    return fig

def show_chatelet_alignement_figure_lines():
    df_viz = pd.read_hdf(r"app/data/chatelet_une_semaine.h5")
    df_viz['date'] = df_viz['datehour'].dt.date
    fig = go.Figure()
    for d in df_viz['date'].unique():
        fig.add_scatter(x=df_viz['hour'], y=df_viz[df_viz.date == d]['delta'], name=f"{d}", mode='markers+lines',)
    fig.update_xaxes(dtick=1, minallowed=-0.5, maxallowed=23.5, title={'text' : 'Heure'})
    fig.update_yaxes(range=[-10, 10])
    fig.update_layout(title="Alignement des pics d'utilisation de la station Chatelet pour la 7ème semaine 2025")
    return fig

@st.cache_resource
def show_chatelet_smoothed():
    df_viz = pd.read_hdf(r"app/data/chatelet_une_semaine.h5")
    fig = go.Figure()
    fig.add_scatter(x=df_viz['datehour'], y=df_viz['delta'], name="Delta")
    fig.add_scatter(x=df_viz['datehour'], y=df_viz['delta_smoothed'], name="Delta Smoothed", line={'color' : 'red'})
    fig.update_layout(title="Lissage de pics d'utilisation de la station Chatelet pour la 7ème semaine 2025 lissage par moyenne glissante.")
    return fig

@st.cache_resource
def show_chatelet_smoothed_alignement_figure():
    df_viz = pd.read_hdf(r"app/data/chatelet_une_semaine.h5")
    fig = go.Figure()
    fig.add_scatter(x=df_viz['hour'], y=df_viz['delta_smoothed'], name="Info d'origine", mode='markers', marker={'color' : df_viz['color']})
    fig.update_xaxes(dtick=1, minallowed=-0.5, maxallowed=23.5, title={'text' : 'Heure'})
    fig.update_yaxes(range=[-10, 10])
    fig.update_layout(title="Alignement des pics d'utilisation de la station Chatelet pour la 7ème semaine 2025 après lissage")
    return fig

@st.cache_resource
def show_chatelet_smoothed_alignement_figure_lines():
    df_viz = pd.read_hdf(r"app/data/chatelet_une_semaine.h5")
    df_viz['date'] = df_viz['datehour'].dt.date
    fig = go.Figure()
    for d in df_viz['date'].unique():
        fig.add_scatter(x=df_viz['hour'], y=df_viz[df_viz.date == d]['delta_smoothed'], name=f"{d}", mode='markers+lines',)
    fig.update_xaxes(dtick=1, minallowed=-0.5, maxallowed=23.5, title={'text' : 'Heure'})
    fig.update_yaxes(range=[-10, 10])
    fig.update_layout(title="Alignement des pics d'utilisation de la station Chatelet pour la 7ème semaine 2025")
    return fig

@st.cache_resource
def show_classic_models_pred():
    df_viz = pd.read_hdf('app/data/models_classics_pred.h5')
    fig = go.Figure()
    for n in ['delta', 'LinearRegression', 'XGBRegressor']:
        fig.add_scatter(x=df_viz.datehour, y=df_viz[n], name=n)
    fig.update_layout(title='Valeurs prédites et réelles (delta) sans lissage pour la station Chatelet.')
    return fig

@st.cache_resource
def show_classic_models_pred_smooth():
    df_viz = pd.read_hdf('app/data/models_classics_pred.h5')
    fig = go.Figure()
    for n in ['smooth delta', 'LinearRegression_smooth', 'XGBRegressor_smooth']:
        fig.add_scatter(x=df_viz.datehour, y=df_viz[n], name=n)
    fig.update_layout(title='Valeurs prédites et réelles (delta) lissées pour la station Chatelet.')
    return fig

def show_history(history_file:str, tag = ''):
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except Exception:
        return None
    fig = go.Figure()
    fig.add_scatter(y=history['loss'], name='loss')
    fig.add_scatter(y=history['val_loss'], name='val_loss')
    fig.update_layout(title=f"Historique d'apprentissage {tag}")
    st.plotly_chart(fig)
    fig = go.Figure()
    fig.add_scatter(y=history['mse'], name='mse')
    fig.add_scatter(y=history['val_mse'], name='val_mse')
    fig.update_layout(title=f"Historique d'apprentissage {tag}")
    st.plotly_chart(fig)

def show_station_pred(dt, delta, test, pred, tag = 'Chatelet'):
    fig = go.Figure()
    fig.add_scatter(x=dt, y=delta, name="delta d'origine", visible='legendonly')
    fig.add_scatter(x=dt, y=test, name="delta test")
    fig.add_scatter(x=dt, y=pred, name="delta pred")
    fig.update_layout(title=f"Flux réel et predit pour la station {tag}")
    return fig

def show():
    st.write(
"""
# Sélection de modèles
## Challenges
Dans le cadre d'analyse de comportement des modèles prédictifs nous avons dû considérer un nombre de traitements visant à minimiser les problèmes principaux de nos données.

### Pics de flux

Il s'agit notamment des pics de flux qui représentés des valeurs extrêmes et assez rares pour être considéré comme aberrantes. On marque ces enregistrement comme "rush" pour les heures de pointe.

```Pour la période entre Dec'2024 et Mar'2025 on a 9.35% de flux > 4 et < -4 qui représentent respectivement Q2 + 1.5 * IQR et Q1 - 1.5 * IQR.```

En réalité se sont les pics d'utilisation qu'on cherche à prédire: les vagues de prises et de retours massifs de vélos aux stations. On veux savoir à quel moment les utilisateurs sont censés de prendre beaucoup de vélos quand les fluctuations minima d'utilisation normale ne porte pas un tel intérêt pour nous.

Pour balancer le dataset on a prévu plusieurs approches:
- Overfitting, multiplication artificielle de class "rush" dans le dataset d'entrainement.
- Underfitting: resampling de la classe majoritaire pour qu'elle soit seulement 3 fois plus grande que le class "rush".
- Weighting: pour les modèles qui le supporte, matrice de multiplicateurs pour attribuer beaucoup plus d'importance au perte sur les enregistrements "rush".
"""
    )
    st.write(
"""
### Heures de pointe flottantes
Les pics mentionnés ne sont pas assez ponctuels comme on peut le constater dans l'exemple suivant. On voit que les prises et les retours de vélos massive peuvent arriver une heure plus tôt ou plus tard dans une station. C'est critique pour les évaluations de prédiction et les calcules de pertes. Si le modèle prédit correctement le pic mais en réalité ce jour-là il à été enregistré une heure plus tard, on aura deux erreur importantes en place d'un bon score.
""")
    st.plotly_chart(show_chatelet_alignement_figure())    
    st.write(
"""
Cette spécificité est importante à prendre en compte pendant le traitement de résultats de prédiction. La comparaison des distributions de valeurs prédites et réelles, ainsi que la visualisation des deux flux, sont donc un ajout indispensable aux calculs d'erreurs.

Pour gérer ce problème, on peut "lisser" les courbes avec une moyenne à fenêtre flottante de 3 valeurs. Cela permettra de calculer le nombre de vélos ou de places libres nécessaires pour la station donnée au cours de quelques heures, tout en diminuant le choc dû au non-alignement horaire.
""")
    st.plotly_chart(show_chatelet_smoothed())
    st.plotly_chart(show_chatelet_smoothed_alignement_figure())
    st.write(
"""
## Modèles classiques

On a testé une série de modèles classique, des séries temporelle et des réseaux neuronaux. Un dataset en versions sans et avec lissage a servi pour l'entrainement. Sans surprise des modèles plus complexes ont démontré une meilleure performance.
"""
    )
    st.dataframe(pd.read_hdf('app/data/models_classics_0.h5'))
    st.write(
"""
Malgré les erreurs relativement faibles ses modèles ne captent pas assez le dynamique de flux de vélos.
"""
    ) 
    st.plotly_chart(show_classic_models_pred())
    
    st.plotly_chart(show_classic_models_pred_smooth())
    
    st.write(
"""
## Modèle Prophet pour la station Châtelet

Nous avons également testé le modèle Prophet, une méthode de prévision de séries temporelles développée par Facebook, sur les données de la station Châtelet. Ce modèle est particulièrement adapté aux séries temporelles avec des tendances fortes et des saisonnalités multiples.

Cependant, malgré sa capacité à gérer les composantes saisonnières, les résultats obtenus pour la station Châtelet n'ont pas été satisfaisants. L'image suivante montre les performances du modèle Prophet, où l'on peut observer que les prédictions ne capturent pas correctement les variations réelles des données, en particulier les pics d'utilisation.

""")
    st.image("app/data/perf_prophet_chatelet.png", caption="Performance du modèle Prophet pour la station Châtelet")
    st.write(
"""
Les limites observées peuvent être attribuées à la nature spécifique de nos données, notamment les heures de pointe flottantes et les variations soudaines qui ne suivent pas toujours des patterns saisonniers réguliers. Cela suggère que Prophet, bien qu’efficace pour des séries plus stables, n’est pas optimal pour ce cas d’utilisation sans ajustements supplémentaires significatifs.

## Réseaux neuronaux

La structure complèxe de données et leur volume propose l'utilisation de modèles MLP et CNN pour la prédiction de valeurs.

Suite à nos testes les paramètres suivant de prétraitement de données et de modèles ont été sélectionnées pour les réseaux neuronaux:

- Lissage par un moyen flottant avec la taille de fenêtre 3.
- Pas de suréchantillonnage.
- Fonction d'activation 'tanh'.
- Une seule valeur de poids (sample weight) égale à 5 pour les 'deltas' négatifs et positifs dans le calcule de pertes.
  
> On utilise une fonction de loss personnalisée qui multiplie l'erreur moyenne absolue **MAE** par **5** pour les heures de pointes détectées selon la valeur de 'delta' réel.

### MultiLayer Perceptron MLP

Un modèle *Sequential* de couches *Dense* qui prédit une valeur 'delta' à la base d'une ligne de features.

> Une instance de données pour ce modèle (num_features,) est un échantillon de notre dataset: un enregistrement pour une station donnée (coordonnées géographiques, cluster et capacité) au moment donné (heure, jour de la semaine, jour férié etc.) avec les informations météo actuelles.

Le modèle à été compilé avec l'optimisateur **'adam'**, il utilise 60 801 paramètres et à été entrainé en 200 époques.
"""
    )
    show_history('app/data/mlp_fit_hist.json', 'MLP smoothed weighted')

    st.write(
"""
**Scores:**
- *RMSE*: 1.65
- *MAE*: 1.11
- *RMSE pics*: 3.18
- *MAE pics*: 2.47
"""
    )
    chatelet = '82328045'
    lefebvre = '3906215030'
    
    df = pd.read_hdf('app/data/mlp_best_pred.h5')
    df_viz = df[df.station == chatelet]
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred']))
    # df_viz = df[df.station == lefebvre]
    # st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred'], tag = df_viz['name'].iloc[0] + ' (cas problématique)'))

    st.write(
"""    
### Réseaux neuronaux convolutif CNN

Un modèle base de couches Conv1D et Dense qui prédit **24** valeur 'delta' à la base d'une **table 2d** de features.

> Chaque instance de données (24, num_features) représente les 24 échantillons pour chaque heure d'une journée donnée pour une station donnée (batch de dataset regroupé par station et date). Dans le dataset d'entrainement il y a 99 456 instances soit 1344 stations x 74 jours.

Le modèle à été compilé avec l'optimisateur **'adam'**, il utilise 389 272 paramètres et à été entrainé en 42 époques (arrêté par *EarlyStopping*).

"""
    )
    show_history('app/data/cnn_fit_hist.json', 'CNN smoothed weighted')

    st.write(
"""
**Scores:**
- *RMSE*: 1.76
- *MAE*: 1.19
- *RMSE pics*: 3.36
- *MAE pics*: 2.73
"""
    )
    chatelet = '82328045'
    lefebvre = '3906215030'
    
    df_cnn = pd.read_hdf('app/data/cnn_best_pred.h5')
    df_viz = df_cnn[df_cnn.station == chatelet]
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred']))
    # df_viz = df[df.station == lefebvre]
    # st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred'], tag = df_viz['name'].iloc[0] + ' (cas problématique)'))

    st.write(
"""
# Top-3
## XGB Performant
Les modèles MLP et CNN présenté ont été les plus performants au cours de testes. Pour bien finaliser le top-3 il vaut noter le **XGBRegressor** qui avec un lissage de données sur une fenêtre de **4** et weighting égale à **5** est capable pour des meilleurs prédictions:

**Scores:**
- *RMSE*: 1.4
- *MAE*: 1.01
- *RMSE pics*: 3.05
- *MAE pics*: 2.62
"""
    )
    df_xgb = pd.read_hdf('app/data/xgb_best_pred.h5')
    df_viz = df_xgb[df_xgb.station == chatelet]
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred']))

    st.write(
"""

## Problème d'application
Les modèles Top-3 permettent de savoir avec une marge raisonnable combien de vélos ou de places libres il faut avoir pendant les heures de pointes le matin et le soir sur des stations comme Chatelet.
"""
    )
    pred_bike_counts = pd.read_hdf('app/data/pred_bike_counts.h5')
    st.dataframe(pred_bike_counts, hide_index=True)
    st.write(
"""
Mais on constate également que si la prédiction est correcte pour Chatelet ce n'est pas toujours le cas. Même si la sélection de modèle s'appuyait déjà sur ce facteur, au niveau globale les modèles les plus performants montre un bon potentiel pour améliorations. On peut distinguer un nombre de stations problématiques. Voici le cas de MLP:
"""
    )
    df_viz = df[df.station == lefebvre]
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred'], tag = df_viz['name'].iloc[0] + ' (cas problématique MLP)'))
    df_viz = df[df.station == '653205656']    
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred'], tag = df_viz['name'].iloc[0] + ' (cas problématique MLP)'))

    st.write(
"""
Ces problèmes se manifestent dans les erreurs calculées par station:
"""
    )
    mlp_stations_scores = pd.read_hdf('app/data/mlp_stations_scores.h5')
    st.plotly_chart(px.box(mlp_stations_scores[['rmse', 'pic_rmse']], orientation='h', title='Distribution de RMSE et RMSE sur les heure de pointe (pic) pour les stations (MLP)', points="suspectedoutliers"))
    st.dataframe(mlp_stations_scores)
    st.write(
"""
Pour diminuer la disparité des stations en prédiction on pourrait envisager la séparation de dataset en deux partie selon l'efficacité de modèles actuelle au niveau local. Il y a encore du potentiel pour un réglage plus fin de modèle CNN qui permet à estimer les relations entre les valeurs 'delta' consécutives.
Il serait également possible, avec plus de pouvoir matériel de calcul, d'élaborer une structure multidimensionnelle alternative pour séparer explicitement les entrainement et les prédictions par station.
"""
    )