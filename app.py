import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from decompose_ts import decompose_ts
from selector import selector
from acf_pacf import acf_pacf
from grid import grid
from plot import plot
from predict import predict
from sidebarmenu import sidebarmenu
from stationarity import stationarity
from train import train
from transform import transform

pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.image('img/8.png')
st.write('> *«Скажите, что произойдет в будущем, и мы будем знать, что вы боги»  \n(Ис 41:23)*')
st.markdown('**Cервис для автоматизированного прогноза временных рядов. Файлы с данными  \nзагружаются в форматах'
            ' *.csv .txt .xls .xlsx *. Необходимые настройки указыватся в боковом меню слева.**')

st.sidebar.title('Конфигурация данных')

uploaded_file = st.sidebar.file_uploader("Загрузите файл в подходящем фомате ",type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    filename = uploaded_file
else:
    filename, df = selector()

st.header('Предварительный просмотр данных')
st.dataframe(df.head(10))

ds_column, y, data_frequency, test_set_size, exog_variables = sidebarmenu('feature_target', df=df)

exog_variables_names = exog_variables

exog_variables = df[exog_variables] if len(exog_variables) > 0 else None

plot_menu_title = st.sidebar.markdown('### Графики')
plot_menu_text = st.sidebar.text('Выберите, какие нужно отобразить')
show_absolute_plot = sidebarmenu('Временной ряд')
show_seasonal_decompose = sidebarmenu('Сезонная декомпозиция')
show_adfuller_test = sidebarmenu('Дики-Фуллер')
show_train_prediction = sidebarmenu('Прогноз по обучающей выборке')
show_test_prediction = sidebarmenu('Прогноз по тестовой выборке')
force_transformation = sidebarmenu('Принудительная трансформация')

difference_size = None
seasonal_difference_size = None

if 'Пользовательские параметры' in force_transformation:
    difference_size = st.sidebar.slider('Количество дифференцирований: ', 0, 30, 1)
    seasonal_difference_size = st.sidebar.slider('Количество сезонных дифференцирований: ', 0, 30, 1)

plot_adfuller_result = False
if show_adfuller_test:
    plot_adfuller_result = True

df = transform(df, ds_column, data_frequency, y)

if show_absolute_plot:
    st.header('Временной ряд')
    df[y].plot(color='green')
    plt.title('Исходный временной ряд')
    st.pyplot()

if show_seasonal_decompose:
    st.header('Сезонная декомпозиция')
    decompose_ts(df)

st.header('Проверка ряда на стационарность')

if force_transformation == None:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = \
        stationarity(df[y], plot_adfuller_result, data_frequency)
else:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = \
        stationarity(df[y], plot_adfuller_result, data_frequency, force_transformation_technique = force_transformation,
                     custom_transformation_size = (difference_size, seasonal_difference_size))

st.header('Функции автокорреляции и частичной автокорреляции (ACF/PACF)')
p, q, P, Q = acf_pacf(acf_pacf_data, seasonality)
st.markdown('*Начальное приближение по ACF/PACF: {}x{}{}*'.format((p, d, q), (P, D, Q), (seasonality)))

st.header('Обучить модель')
st.write('Выберите подходящие параметры в меню слева и нажмите на кнопку')

try:
    p, d, q, P, D, Q, s, train_model, periods_to_forecast, execute_grid_search = \
        sidebarmenu('terms', test_set_size, seasonality, (p, d, q, P, D, Q, seasonality), df=ts)
except ValueError:
    error_message = '''
                    Произошла ошибка при нахождении начального приближения для параметров.
                    Убедитесь, что частота данных указана верно.
                    '''
    raise ValueError(error_message)

if execute_grid_search:
    if data_frequency in ['Часовая', 'Дневная'] or p >= 5 or q >= 5:
        warning_grid_search = '''
                            Перебор парметров будет вычислительно сложным и может занять время.
                            '''
        st.sidebar.warning(warning_grid_search)

if train_model:
    exog_train = None
    exog_test = None

    if type(exog_variables) == type(pd.DataFrame()):
        exog_variables.index = ts.index
        exog_train = exog_variables.iloc[:-test_set_size]
        exog_test = exog_variables.iloc[-test_set_size:]

    train_set = transformation_function(ts.iloc[:-test_set_size])
    
    test_set = transformation_function(ts.iloc[-test_set_size:])
    
    try:
        model = train(train_set, p, d, q, P, D, Q, s, exog_variables=exog_train, quiet=False)
    except ValueError as ve:
        if ve.args[0] == 'maxlag should be < nobs':
            raise ValueError('Похоже, файл имеет мало данных. Укажите меньшие значения для (p, q, P, Q)')
        else:
            raise ve

    st.markdown('## **Прогноз по обучающей выборке**')
    if transformation_function == np.log1p:
        predict(train_set.iloc[-24:], y, seasonality, np.expm1, model, show_train_prediction =
        show_train_prediction, show_test_prediction=show_test_prediction)
    else:
        predict(train_set.iloc[-24:], y, seasonality, transformation_function, model, show_train_prediction =
        show_train_prediction, show_test_prediction=show_test_prediction)
    
    st.markdown('## **Прогноз по тестовой выборке**')
    if transformation_function == np.log1p:
        predict(test_set, y, seasonality, np.expm1, model, exog_variables = exog_test,forecast=True,
                show_train_prediction = show_train_prediction, show_test_prediction = show_test_prediction)
    else:
        predict(test_set, y, seasonality, transformation_function, model, exog_variables = exog_test, forecast=True,
                show_train_prediction = show_train_prediction, show_test_prediction = show_test_prediction)

    if execute_grid_search:
        st.markdown('# Поиск параметров перебором')
        st.markdown('''
                    Поиск наилучших параметров займет время.
                    Ожидайте...
                    ''')
        p1, d1, q1, P1, D1, Q1, s = grid(train_set, exog_train, range(p+2), range(q+2), range(P+2), range(Q+2),
                                         d=d, D=D, s=s)

    st.markdown('# Прогноз вне выборки')

    with st.spinner('Обучение модели на всей выборке. Ожидайте...'):
        final_model = train(transformation_function(ts), p, d, q, P, D, Q, s, exog_variables=exog_variables, quiet=True)
    st.success('Выполнено!')
    
    if type(exog_variables) == type(pd.DataFrame()):
        st.write('Выбраны экзогенные переменные. Прогноз не может быть выполнен из-за недостаточного количества'
                 ' переменных' )
    else:
        if transformation_function == np.log1p:
            forecasts = np.expm1(final_model.forecast(periods_to_forecast))
            confidence_interval = np.expm1(final_model.get_forecast(periods_to_forecast).conf_int())
            

        else:
            forecasts = final_model.forecast(periods_to_forecast)
            confidence_interval = final_model.get_forecast(periods_to_forecast).conf_int()

        confidence_interval.columns = ['ДИ нижний', 'ДИ верхний']
        plot(forecasts, confidence_interval, data_frequency)

