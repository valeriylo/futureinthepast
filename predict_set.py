import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from math import sqrt
from mean_abs_pct_error import mean_abs_pct_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict_set(timeseries, y, seasonality, transformation_function, model, exog_variables=None,forecast=False, show_train_prediction=None, show_test_prediction=None):

    timeseries = timeseries.to_frame()
    timeseries[y] = transformation_function(timeseries[y])

    if forecast:
        timeseries['ŷ'] = transformation_function(model.forecast(len(timeseries), exog=exog_variables))
    else:
        timeseries['ŷ'] = transformation_function(model.predict())
    
    if show_train_prediction and forecast == False:
        timeseries[[y, 'ŷ']].iloc[-(seasonality*3):].plot(color=['green', 'red'])

        plt.ylabel(y)
        plt.xlabel('')
        plt.title('Прогноз по обучающей выборке')
        st.pyplot()
    elif show_test_prediction and forecast:
        timeseries[[y, 'ŷ']].iloc[-(seasonality*3):].plot(color=['green', 'red'])

        plt.ylabel(y)
        plt.xlabel('')
        plt.title('Прогноз по тестовой выборке')
        st.pyplot()

    try:
        rmse = sqrt(mean_squared_error(timeseries[y].iloc[-(seasonality*3):], timeseries['ŷ'].iloc[-(seasonality*3):]))
        aic = model.aic
        bic = model.bic
        hqic = model.hqic
        mape = np.round(mean_abs_pct_error(timeseries[y].iloc[-(seasonality*3):], timeseries['ŷ'].iloc[-(seasonality*3):]), 2)
        mae = np.round(mean_absolute_error(timeseries[y].iloc[-(seasonality*3):], timeseries['ŷ'].iloc[-(seasonality*3):]), 2)
    except ValueError:
        error_message = '''
                        Возникли проблемы с расчетом метрик модели.
                        Обычно это происходит из-за неверного формата данных в столбце Даты.
                        Убедись в соответствии формата с форматом функции Pandas to_datetime.
                        '''
        raise ValueError(error_message)
    
    metrics_df = pd.DataFrame(data=[rmse, aic, bic, hqic, mape, mae], columns = ['{} SET METRICS'.format('TEST' if forecast else 'TRAIN')], index = ['RMSE', 'AIC', 'BIC', 'HQIC', 'MAPE', 'MAE'])
    st.markdown('### **Метрики**')
    st.dataframe(metrics_df)
