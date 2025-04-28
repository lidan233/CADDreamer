from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pygal
from IPython.display import SVG, display



def plot_line(chartname,name,X,Y):
    if len(name)!=len(Y):
        raise ValueError('name and Y in plotline has different length.')
    line_chart = pygal.Line()
    line_chart.title = chartname
    line_chart.x_labels = X
    for i in range(len(name)):
        line_chart.add(name[i], Y[i])
    line_chart.render_in_browser()

def plot_line_browser(chartname,name,X,Y,xlabel='',ylabel=''):
    fig  = go.Figure()
    for i in range(len(Y)):
        fig.add_trace(go.Scatter(x=X, y=Y[i], name=name[i],line=dict( width=4)))
    fig.update_layout(title=chartname,
                      xaxis_title=xlabel,
                      yaxis_title=ylabel)
    fig.show()

def plot_bar_browser(chartname,name,X,Y,xlabel='',ylabel=''):


    bars = []
    colors = []

    for x in X:
        if x == 'Our method':
            colors.append('crimson')
        else:
            colors.append('blue')

    for i in range(len(Y)):
        bars.append(go.Bar(x=X, y=Y[i], name=name[i],marker_color=colors))
    fig = go.Figure(data=bars)
    fig.update_layout(title=chartname,
                      xaxis_title=xlabel,
                      yaxis_title=ylabel,
                      font=dict(size=20),
                        yaxis_exponentformat ='E' ,
                        barmode = 'group')
    fig.show()


def matplotlib_plotline(chartname,name,X,Y):
    plt.title(chartname)
    for i in range(len(Y)):
        plt.plot(X, Y[i],label=name[i])
    plt.show()


