from numpy import isscalar
from pandas import DatetimeIndex, date_range, merge
from streamlit import warning
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm

def test(ts):
    '''
    Элементарная проверка для корректной работы SARIMAX
    '''
    seasonal_decompose(ts)

    mod = sm.tsa.statespace.SARIMAX(ts, order = (0, 0, 1))
    results = mod.fit()

    assert not isscalar(results.forecast(10).index[0]), 'The forecasts index is not a datetime type'

def transform(df, ds_column, date_frequency, y):
    '''
    Предварительная обработка данных
    '''

    date_frequency_dict = {'Часовая': 'ч',
                           'Дневная': 'д',
                           'Ежемесячная': 'м',
                           'Квартальная': 'к',
                           'Годовая': 'г'}
    
    df.set_index(ds_column, inplace = True)
    df = df.dropna()
    try:
        df.index = df.index.astype('datetime64[ns]')
        test(df[y])
    except:
        try:
            date_format = DatetimeIndex(df.index[-10:], freq='infer')

            df.index = df.asfreq(date_format.freq, fill_value=0)
            test(df[y])
        except ValueError:
            try:
                fill_date_range = date_range(df.index.min(), df.index.max(), freq=date_format.freq)
                df = merge(fill_date_range.to_frame().drop(0, axis=1), 
                           df, 
                           how = 'left', 
                           right_index = True, 
                           left_index = True)
                null_values = df[df.loc[:, y].isnull()].index.values
                if len(null_values) > 0:
                    warning('Найдены пустые значения. Они были заполнены нулями.'.format(null_values))
                    df = df.fillna(0)
                test(df[y])
            except:
                try:
                    warning_message = '''
                                Возникла проблема при определения частоты ряда. 
                                Произведена попытка приведения к частоте указанной в меню конфигурации, однако
                                необходимо убедиться, что данные частотности, указанной в меню выбора 
                                (Часовая, Дневная, Ежемесячная, Квартальная, Годовая)
                                '''
                    warning(warning_message)
                    df = df.asfreq(date_frequency_dict[date_frequency])

                    null_values = df[df.loc[:, y].isnull()].index.values
                    if len(null_values) > 0:
                        warning('Найдены пустые значения. Они были заполнены нулями.'.format(null_values))
                        df = df.fillna(0)
                    test(df[y])
                except:
                    error_message = '''
                                    Возникла проблема при конвертировании столбца с датой.
                                    Убедитесь, что значения даты соответствуют формату функции to_datetime function
                                    пакета Pandas. Подробнее 
                                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
                                    '''
                    raise TypeError(error_message)
    return df
