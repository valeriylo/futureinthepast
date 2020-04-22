import streamlit as st
import sys

def sidebar_menus(menu_name, test_set_size=None, seasonality=None, terms=(0, 0, 0, 0, 0, 0, 0), df=None):
    '''
    Generates the sidebar menus for Streamlit. Based on menu_name parameter, it returns different menus for each situation.
    Args.
        menu_name (str): a menu name that will be shown on the sidebar. It can be: absolute, seasonal, adfuller, train_predictions,
        test_predictions, feature_target, or terms
        seasonality (str, optional): a value to be replaced by a number. e.g.: if Hourly, this function will consider 24 for seasonality
        terms (7-value tuple): tuple with 7 integer values for p, d, q, P, D, Q, and s
        df (Pandas DataFrame, optional): a Pandas DataFrame containing some time series data to extract the columns
    '''
    seasonality_dict = {'Часовая': 24,
                        'Дневная': 7,
                        'Ежемесячная': 12,
                        'Квартальная': 4,
                        'Годовая': 5}

    if menu_name == 'Временной ряд':
        show_absolute_plot = st.sidebar.checkbox('Временной ряд', value=True)
        return show_absolute_plot
    elif menu_name == 'Сезонная декомпозиция':
        show_seasonal_decompose = st.sidebar.checkbox('Сезонная декомпозиция', value=True)
        return show_seasonal_decompose
    elif menu_name == 'Дики-Фуллер':
        show_adfuller = st.sidebar.checkbox('Тест Дики-Фуллера', value=True)
        return show_adfuller
    elif menu_name == 'Прогноз по обучающей выборке':
        show_train_predict_plot = st.sidebar.checkbox('Прогноз по обучающей выборке', value=True)
        return show_train_predict_plot
    elif menu_name == 'Прогноз по тестовой выборке':
        show_test_predict_plot = st.sidebar.checkbox('Прогноз по тестовой выборке', value=True)
        return show_test_predict_plot
    elif menu_name == 'feature_target':
        data_frequency = st.sidebar.selectbox('Какая периодичность данных? ', ['Выберите периодичность', 'Часовая',
                                                            'Дневная', 'Ежемесячная', 'Квартальная', 'Годовая'], 0)
        
        # If the frequency do not select a frequency for the dataset, it will raise an error
        if data_frequency == 'Выберите периодичность':
            # Hiding traceback in order to only show the error message
            sys.tracebacklimit = 0
            raise ValueError('Пожалуйста, выберите периодичность данных')
        
        # Show traceback error
        sys.tracebacklimit = None

        st.sidebar.markdown('### Выбор переменных')
        ds_column = st.sidebar.selectbox('Выберите столбец со временем', df.columns, 0)
        y = st.sidebar.selectbox('Какую переменную нужно предсказать', df.columns, 1)
        exog_variables = st.sidebar.multiselect('Какие переменные экзогенные?', df.drop([ds_column, y], axis=1).columns)
        test_set_size = st.sidebar.slider('Размер валидационной выборки', 3, 30, seasonality_dict[data_frequency])
        return ds_column, y, data_frequency, test_set_size, exog_variables
    elif menu_name == 'Принудительная трансформация':
        st.sidebar.markdown('### Принудительная трасформация данных (опционально)')
        transformation_techniques_list = ['Выбрать оптимальную', 'Без трансформации', 'Первое дифференцирование', 'Логарифмическое', 'Сезонное', 'Первое логарифмическое', 'Логарифмическое + Сезонное', 'Пользовательские параметры']
        transformation_techniques = st.sidebar.selectbox('Метод трансформации', transformation_techniques_list, 0)
        return transformation_techniques
    elif menu_name == 'terms':
        st.sidebar.markdown('### Параметры модели')
        st.sidebar.text('Значения для (p, d, q)x(P, D, Q)s')
        p = st.sidebar.slider('p (AR)', 0, 30, min([terms[0], 30]))
        d = st.sidebar.slider('d (I)', 0, 3, min([terms[1], 3]))
        q = st.sidebar.slider('q (MA)', 0, 30, min([terms[2], 30]))
        P = st.sidebar.slider('P (Сезонная AR)', 0, 30, min([terms[3], 30]))
        D = st.sidebar.slider('D (Сезонное дифференцирование)', 0, 3, min([terms[4], 3]))
        Q = st.sidebar.slider('Q (Сезонное MA)', 0, 30, min([terms[5], 30]))
        s = st.sidebar.slider('s (Сезонная периодичность)', 0, 30, min([terms[6], 30]))
        
        st.sidebar.markdown('# Период прогноза')
        periods_to_forecast = st.sidebar.slider('На какой срок нужен прогноз?', 1, int(len(df.iloc[:-test_set_size])/3), int(seasonality/2))
        
        grid_search = st.sidebar.checkbox('Подобрать наилучшие параметры')
        train_model = st.sidebar.button('Выполнить')

        return p, d, q, P, D, Q, s, train_model, periods_to_forecast, grid_search
