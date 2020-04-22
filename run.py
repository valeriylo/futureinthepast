import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sys

from decompose_series import decompose_series
from file_selector import file_selector
from find_acf_pacf import find_acf_pacf
from generate_code import generate_code
from grid_search_arima import grid_search_arima
from plot_forecast import plot_forecasts
from predict_set import predict_set
from sidebar_menus import sidebar_menus
from test_stationary import test_stationary
from train_ts_model import train_ts_model
from transform_time_series import transform_time_series

pd.set_option('display.float_format', lambda x: '%.3f' % x)

description =   '''
                Web-сервис прогнозирования временных рядов с помощью авторегрессионных алгоритмов и алгоритмов
                градиентного бустинга. Принимаются таблицы данных в популярных форматах .txt, .csv, .xlsx, .xls .
                При необходимости возможна самостоятельная настройка параметров.
                '''
# Description
st.write('> “Forecasting is the art of saying what will happen, and then explaining why it did\'t!”')
st.write('')
st.write(description)

### SIDEBAR
st.sidebar.title('Ваши данные')

uploaded_file = st.sidebar.file_uploader("Загрузите файл в подходящем фомате ",type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    filename = uploaded_file
else:
    filename, df = file_selector()

st.markdown('## **Предварительные просмотр таблицы данных**')
st.dataframe(df.head(10))

ds_column, y, data_frequency, test_set_size, exog_variables = sidebar_menus('feature_target', df=df)

# Name of the exogenous variables
exog_variables_names = exog_variables

# If there's not exogenous variables, it returns None
exog_variables = df[exog_variables] if len(exog_variables) > 0 else None

# Show plots
plot_menu_title = st.sidebar.markdown('### Графики')
plot_menu_text = st.sidebar.text('Выберите, какие Вы бы хотели построить')
show_absolute_plot = sidebar_menus('Временной ряд')
show_seasonal_decompose = sidebar_menus('Сезонная декомпозиция')
show_adfuller_test = sidebar_menus('Дики-Фуллер')
show_train_prediction = sidebar_menus('Прогноз по обучающей выборке')
show_test_prediction = sidebar_menus('Прогноз по тестовой выборке')
force_transformation = sidebar_menus('Принудительная трансформация') # You can force a transformation technique

difference_size = None
seasonal_difference_size = None

if 'Пользовательские параметры' in force_transformation:
    # If the user selects a custom transformation, enable the difference options
    difference_size = st.sidebar.slider('Количество дифференцирований: ', 0, 30, 1)
    seasonal_difference_size = st.sidebar.slider('Количество сезонных дифференцирований: ', 0, 30, 1)

plot_adfuller_result = False
if show_adfuller_test:
    plot_adfuller_result = True

# Transform DataFrame to a Series
df = transform_time_series(df, ds_column, data_frequency, y)

# Show the historical plot?
if show_absolute_plot:
    st.markdown('# Временной ряд ')
    df[y].plot(color='green')
    plt.title('Исходный временной ряд')
    st.pyplot()

# Show decomposition plot
if show_seasonal_decompose:
    st.markdown('# Сезонная декомпозиция')
    decompose_series(df)

# Checking for stationarity in the series
st.title('Проверка ряда на стационарность')

# If a function is not forced by the user, use the default pipeline
if force_transformation == None:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = test_stationary(df[y], plot_adfuller_result, data_frequency)
else:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = test_stationary(df[y], plot_adfuller_result, data_frequency, 
                                                                                                            force_transformation_technique = force_transformation, 
                                                                                                            custom_transformation_size = (difference_size, seasonal_difference_size))

st.title('Функции автокорреляции и частичной автокорреляции (ACF/PACF)')
p, q, P, Q = find_acf_pacf(acf_pacf_data, seasonality)
st.markdown('**Начальное приближение по ACF/PACF**: {}x{}{}'.format((p, d, q), (P, D, Q), (seasonality)))

st.title('Обучить модель')
st.write('Выберите подходящие параметры в меню слева и нажмите на кнопку')

try:
    p, d, q, P, D, Q, s, train_model, periods_to_forecast, execute_grid_search = sidebar_menus('terms', test_set_size, seasonality, (p, d, q, P, D, Q, seasonality), df=ts)
except ValueError:
    error_message = '''
                    A problem has occurred while we tried to find the best initial parameters for p, d, and q.
                    Please, check if your FREQUENCY field is correct for your dataset. For example, if your dataset
                    was collected in a daily basis, check if you selected DAILY in the FREQUENCY field.
                    '''
    raise ValueError(error_message)

# Showing a warning when Grid Search operation is too expensive
if execute_grid_search:
    if data_frequency in ['Часовая', 'Дневная'] or p >= 5 or q >= 5:
        warning_grid_search = '''
                            Перебор парметров будет вычислительно сложным и может занять время.
                            '''
        st.sidebar.warning(warning_grid_search)

# If train button has be clicked 
if train_model:
    exog_train = None
    exog_test = None

    # Aligning endog and exog variables index, if exog_variables is not null
    if type(exog_variables) == type(pd.DataFrame()):
        exog_variables.index = ts.index
        exog_train = exog_variables.iloc[:-test_set_size]
        exog_test = exog_variables.iloc[-test_set_size:]

    train_set = transformation_function(ts.iloc[:-test_set_size])
    
    test_set = transformation_function(ts.iloc[-test_set_size:])
    
    try:
        model = train_ts_model(train_set, p, d, q, P, D, Q, s, exog_variables=exog_train, quiet=False)
    except ValueError as ve:
        if ve.args[0] == 'maxlag should be < nobs':
            raise ValueError('Seems that you don\'t have enough data. Try to use smaller terms for AR and MA (p, q, P, Q)')
        else:
            raise ve

    st.markdown('## **Прогноз по обучающей выборке**')
    #st.write('The model was trained with this data. It\'s trying to predict the same data')
    if transformation_function == np.log1p:
        predict_set(train_set.iloc[-24:], y, seasonality, np.expm1, model, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    else:
        predict_set(train_set.iloc[-24:], y, seasonality, transformation_function, model, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    
    st.markdown('## **Прогноз по тестовой выборке**')
    #st.write('Unseen data. The model was not trained with this data and it\'s trying to forecast')
    if transformation_function == np.log1p:
        predict_set(test_set, y, seasonality, np.expm1, model, exog_variables=exog_test,forecast=True, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    else:
        predict_set(test_set, y, seasonality, transformation_function, model, exog_variables=exog_test, forecast=True, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)

    # Executing Grid Search
    if execute_grid_search:
        st.markdown('# Поиск параметров перебором')
        st.markdown('''
                    Поиск наилучших параметров займет время. Особенно вычислительно сложно почасовая и ежедневная периодичность данных.
                    Ожидайте...
                    ''')
        p1, d1, q1, P1, D1, Q1, s = grid_search_arima(train_set, exog_train,  range(p+2), range(q+2), range(P+2), range(Q+2), d=d, D=D, s=s)
    #grid_final_model = train_ts_model(transformation_function(ts), p1, d1, q1, P1, D1, Q1, s, exog_variables=exog_variables, quiet=True)
    # Forecasting data
    st.markdown('# Прогноз вне выборки')
    
    # Creating final model
    with st.spinner('Обучение модели на всей выборке. Ожидайте.'):
        final_model = train_ts_model(transformation_function(ts), p, d, q, P, D, Q, s, exog_variables=exog_variables, quiet=True)
    st.success('Выполнено!')
    
    if type(exog_variables) == type(pd.DataFrame()):
        st.write('Выбраны экзогенные переменные. Прогноз не может быть выполнен из-за недостаточного количества переменных' )
    else:
        if transformation_function == np.log1p:
            forecasts = np.expm1(final_model.forecast(periods_to_forecast))
            confidence_interval = np.expm1(final_model.get_forecast(periods_to_forecast).conf_int())
            

        else:
            forecasts = final_model.forecast(periods_to_forecast)
            confidence_interval = final_model.get_forecast(periods_to_forecast).conf_int()
            st.write(confidence_interval)

        confidence_interval.columns = ['ДИ нижний', 'ДИ верхний']
        plot_forecasts(forecasts, confidence_interval, data_frequency)

    '''st.write('# Исходный код')
    st.markdown(generate_code(filename, ds_column, y, test_stationarity_code, test_set_size, 
                              seasonality, p, d, q, P, D, Q, s, exog_variables_names, transformation_function, 
                              periods_to_forecast, data_frequency))'''
