import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from transform_f import timeSeriesTransformer

def stationarity(timeseries, plot_results=False, data_frequency=None, force_transformation_technique=None,
                 custom_transformation_size=None):
    '''
   Функция, проверяющая на стационарность с помощью теста Дики-Фуллера
    '''

    transformer = timeSeriesTransformer(timeseries, data_frequency)
    best_transformation = transformer.test_absolute_data()
    progress_bar = st.progress(0)

    if force_transformation_technique != None and force_transformation_technique != 'Выбрыть оптимальную':
        if force_transformation_technique == 'Без трансформации':
            best_transformation = transformer.test_absolute_data()
        if force_transformation_technique == 'Первое дифференцирование':
            best_transformation = transformer.test_first_difference()
        if force_transformation_technique == 'Первое логарифмическое':
            best_transformation = transformer.test_log_difference()
        if force_transformation_technique == 'Логарифмическое':
            best_transformation = transformer.test_log_transformation()
        if force_transformation_technique == 'Сезонное':
            best_transformation = transformer.test_seasonal_difference()
        if force_transformation_technique == 'Логарифмическое + Сезонное':
            best_transformation = transformer.test_seasonal_log_difference()
        if force_transformation_technique == 'Пользовательские параметры':
            if custom_transformation_size == None:
                raise ValueError('Нельзя задать пустое значение для количества дифференцирования')
            best_transformation = transformer.test_custom_difference(custom_transformation_size)

            if best_transformation[2] is None:
                warn_message = '''
                            Транформация статистически незначима. 
                            Тест Дики-Фуллера {:.3f}, и критическое значение в 1% {:.3f}
                            '''.format(best_transformation[0][0], best_transformation[0][4]['1%'])
                st.warning(warn_message)
        progress_bar.progress(100)
    
    else:
        absolute_test = transformer.test_absolute_data()
        progress_bar.progress(20)
        first_difference_test = transformer.test_first_difference()
        progress_bar.progress(40)
        log_difference_test = transformer.test_log_difference()
        progress_bar.progress(60)
        log_transformation_test = transformer.test_log_transformation()
        progress_bar.progress(80)
        seasonal_difference_test = transformer.test_seasonal_difference()
        progress_bar.progress(100)
        seasonal_log_difference_test = transformer.test_seasonal_log_difference()

        transformations = [absolute_test, first_difference_test, log_difference_test, 
                        log_transformation_test, seasonal_difference_test, seasonal_log_difference_test]

        best_transformation = absolute_test

        for transformation in transformations:
            if transformation[0] < best_transformation[0] and transformation[2] != None:
                best_transformation = transformation

    mean = best_transformation[1].rolling(window=best_transformation[7]).mean()
    std = best_transformation[1].rolling(window=best_transformation[7]).std()
    
    if plot_results:
        fig = plt.figure(figsize=(10, 5))
        orig = plt.plot(best_transformation[1], color='green', label='Исходные данные')
        mean = plt.plot(mean, color='red', label='Среднее')
        std = plt.plot(std, color='black', label='Оценка СО')
        plt.legend(loc='best')
        plt.title('Скользящее среднее и Стандартное отклонение')
        st.pyplot()

    st.write('тест Дики-Фуллера')
    stat_test_value = best_transformation[0][0]
    critical_value_1_perc = best_transformation[0][4]['1%']

    dfoutput = pd.Series(best_transformation[0][0:4], index=['Статистический тест', 'p-значение', 'Число лагов',
                                                             'Число наблюдений'])
    for key, value in best_transformation[0][4].items():
        dfoutput['Критическое значение {}'.format(key)] = value
    st.write(dfoutput)

    return timeseries, best_transformation[3], best_transformation[4], best_transformation[7], best_transformation[1], \
           best_transformation[5], best_transformation[6]

