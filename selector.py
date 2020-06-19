import os
import pandas as pd
import streamlit as st

def selector(folder_path='datasets/'):
    '''
    Функция выбора файла пользователя и начальной проверки данных
    '''
    filenames = os.listdir(folder_path)
    filenames.sort()
    default_file_index = filenames.index('monthly_air_passengers.csv') if 'monthly_air_passengers.csv' in filenames \
        else 0
    selected_filename = st.sidebar.selectbox('Выберите файл', filenames, default_file_index)

    if str.lower(selected_filename.split('.')[-1]) in ['csv', 'txt']:
        try:
            df = pd.read_csv(os.path.join(folder_path, selected_filename))
        except pd._libs.parsers.ParserError:
            try:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), delimiter=';')
            except UnicodeDecodeError:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), delimiter=';', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), encoding='latin1')
            except pd._libs.parsers.ParserError:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), encoding='latin1', delimiter=';')

    elif str.lower(selected_filename.split('.')[-1]) == 'xls' or str.lower(selected_filename.split('.')[-1]) == 'xlsx':
        try:
            df = pd.read_excel(os.path.join(folder_path, selected_filename))
        except pd._libs.parsers.ParserError:
            try:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), delimiter=';')
            except UnicodeDecodeError:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), delimiter=';', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), encoding='latin1')
            except pd._libs.parsers.ParserError:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), encoding='latin1', delimiter=';')
    else:
        st.error('Данный формат не поддерживается')

    if len(df) < 30:
        data_points_warning = '''
                              В таблице слишком мало данных для прогнозирования.
                              Рекомендуется таблица данных хотя бы с 50 строчками, оптимально 100 строк.
                              В противном случае, возможны серьезные погрешности.
                              '''
        st.warning(data_points_warning)
    return os.path.join(folder_path, selected_filename), df
