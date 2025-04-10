import streamlit as st
import plotly.graph_objects as go
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
# Séléction de modèles
## Challenges
Dans le cadre d'analyse de comportement des modèles prédictifs nous avons dû considerer un nombre de traitements visants à minimiser les problèmes principaux de nos données.

### Pics de flux

Il s'agit notamment des pics de flux qui répresentes des valeurs extrèmes et assez rares pour être considéré comme abérrantes. On marque ces enregistrement comme "rush" pour heure de pointe.

```Pour la période entre Dec'2024 et Mar'2025 on a 9.35% de flux > 4 et < -4 qui representent respectivement Q2 + 1.5 * IQR et Q1 - 1.5 * IQR.```

Mais en rélité se sont les pics d'utilisation qu'on cherche à prédire: les vagues de prises et de retours massifs des vélos aux stations. On veux savoir à quel moment les utilisateurs sont censés de prendre beaucoup de vélo quand les fluctiations minima d'utilisation normale ne porte pas un tel intérêt pour nous.

Pour balancer le dataset on a prévu plusieurs approches:
- Overfitting, multiplication artificièle de class "rush" dans le dataset d'entrainement.
- Underfitting: resampling de la classe majoritaire pour qu'elle soit seulement 3 fois plus grande que le class "rush".
- Weighting: pour les modèles qui le supporte, matrice de multiplicateurs pour attribuer beaucoup plus d'importance au perte sur les enregistrements "rush".
"""
    )
    st.write(
"""
### Heures de pointe flotantes
Les pics mentionnés ne sont pas assez ponctuels comme on peut le constater dans l'exemple suivant. On voit que les prises et les retours de vélos massive peuvent arriver une heure plus tôt ou plus tard dans une station. C'est critique pour les évaluations de prédiction et les calcules de pertes. Si le modèle prédicte correctement le pic mais en réalité ce jour-là il à été enregistré une heure plus tard, on aura deux erreur importantes en place d'un bon score.
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

On a testé classique, timeseries et rn.
Pour les testes qui ont généré la table comparative ont a utiliser les modèles avec des paramètre par défaut.
"""
    )
    st.dataframe(pd.read_hdf('app/data/models_classics_0.h5'))
    st.write(
"""
Malgré les erreurs relativement faibles les modèles ne captent pas assez le dynamique.
"""
    ) 
    st.plotly_chart(show_classic_models_pred())
    
    st.plotly_chart(show_classic_models_pred_smooth())
    
    st.write(
"""
## Reseaux neuronaux

La structure complèxe de données et leur volume propose l'utilisation de modèles MLP et CNN pour la prédiction de valeurs.

Suite à nos testes les paramètres suivant de prétraitement de données et de modèles ont été séléctionnées pour les réseaux neuronnes:

- Lissage par un moyen flotant avec la taille de fenêtre 3.
- Pas de surechantillonage.
- Fonction d'activation 'tanh'.
- Une seule valeur de poid (sample weight) égale à 5 pour les 'deltas' négatifs et positifs dans le calcule de pertes.
  
> On utilise une fonction de loss personnalisée qui multiplie l'erreur moyenne absolue par 5 pour les heures de pointes detectées selon la valeur de 'delta' réel.

### MultiLayer Perceptron MLP

Un modèle *Sequential* de couches *Dense* qui prédit une valeur 'delta' à la base d'une ligne de features.

> Une instance de données pour ce modèle (num_features,) est un echantillon de notre dataset: un enregistement pour une station donnée (coordonnées géographiques, cluster et capacity) au moment donné (hour, weekday, holiday etc.) avec les informations météo actuelles.

Le modèle à été compilé avec l'optimizateur **'adam'**, il utilise 60 801 paramètres et à été entrainé en 200 epoches.
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
    df_viz = df[df.station == lefebvre]
    st.plotly_chart(show_station_pred(df_viz['datehour'], df_viz['delta'], df_viz['delta_test'], df_viz['pred'], tag = df_viz['name'].iloc[0] + ' (cas problématique)'))

    df = pd.read_hdf('app/data/pred_bike_counts.h5')
    st.dataframe(df[df.Tag == 'MLP'].drop(columns=['Tag']), hide_index=True)


    st.write(
"""    
### Réseau neuronal convolutif CNN

Un modèle base de couches Conv1D et Dense qui prédit *24 valeur* 'delta' à la base d'une table 2d de features.

> Chaque instance de données (24, num_features) représente les 24 echantillons pour chaque heure d'une journée donnée pour une station donnée (batch de dataset regrouppé par station et date).
"""
    )
    st.image('app/data/image_tables_sketch.png')

    
