import plotly.graph_objs as go
import streamlit as st

def plot(forecasts, confidence_interval, periods):
    '''
    Функция строит график прогноза модели
    '''
    lower_ci = {"x": confidence_interval.index, 
                "y": confidence_interval['ДИ нижний'],
                "line": {
                    "color": "#1EBC61", 
                    "shape": "linear",
                    "width": 0.1
                        }, 
                "mode": "lines",
                "name": "Менее 95%",
                "showlegend": False, 
                "type": "scatter", 
                "xaxis": "x", 
                "yaxis": "y"
                }
    upper_ci = {"x": confidence_interval.index, 
                "y": confidence_interval['ДИ верхний'],
                "fill": "tonexty", 
                "line": {
                    "color": "#1EBC61", 
                    "shape": "linear",
                    "width": 0.1
                        }, 
                "mode": "lines", 
                "name": "Более 95%",
                "type": "scatter", 
                "xaxis": "x", 
                "yaxis": "y"
                }
    forecasting =  {'x': forecasts.index, 
                    'y': forecasts.values,
                    "line": {
                            "color": "#005C01", 
                            "shape": "linear",
                            "width": 3
                            }, 
                    "mode": "lines", 
                    "name": "Прогноз",
                    "type": "scatter", 
                    "xaxis": "x", 
                    "yaxis": "y"                }

    plot_data = ([lower_ci, upper_ci, forecasting])
    layout = go.Layout(title = 'Прогноз')
    fig = go.Figure(data = plot_data, layout=layout)
    st.plotly_chart(fig)
