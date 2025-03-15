import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

class VelibDataViz:
    default_task = {
        'x' : 'datehour',
        'y' : 'delta',
        'name' : None,
        'color' : None,
        'fill' : None,
        # 'fill' : 'rgba(80, 0, 0, 0.2)',
        'secondary_y' : False
    }
    @staticmethod
    def line(data : pd.DataFrame, tasks : list[dict], title : str = None, legend : str = None, xaxis : dict = {}, yaxis : dict = {}, filename = None):
        """Designer un graphique linéaire

        Args:
            data (pd.DataFrame): _description_
            task (dict): liste de paramètres comme VelibDataViz.default_task: [{x, y, name(optional), color(optional), fill(optional)}]
            title (str, optional): _description_. Defaults to None.
            legend (str, optional): _description_. Defaults to None.
            xaxis (dict, optional): _description_. Defaults to {}.
            yaxis (dict, optional): _description_. Defaults to {}.
        """
        secondary_y_active = any([t.get('secondary_y') for t in tasks])
        fig = make_subplots(specs=[[{"secondary_y": secondary_y_active}]])# if secondary_y_active else go.Figure()
        for t in tasks:
            t = VelibDataViz.default_task | t
            fig.add_trace(
            go.Scatter(x=data[t['x']], y=data[t['y']],
                fill='tonexty' if t['fill'] else None,
                fillcolor=t['fill'],
                mode='lines+markers',
                line={'color': t['color']},
                name=t['name']
            ), secondary_y=t['secondary_y'])
        fig.update_layout(
            title = title,
            legend_title_text = legend,
            xaxis = xaxis,
            yaxis = yaxis,
            # width = 800
        )
        if filename:
        # f_name = "".join([x if x.isalnum() else "_" for x in title]) + '.html'
            fig.write_html(filename, auto_open=True)
        else:
            fig.show()



        