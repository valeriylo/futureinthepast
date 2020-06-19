import numpy as np

from statsmodels.tsa.stattools import adfuller

class timeSeriesTransformer:
    def __init__(self, original_timeseries, data_frequency):
        self.seasonality_dict = {'Часовая': 24,
                                 'Дневная': 7,
                                 'Ежемесячная': 12,
                                 'Квартальная': 4,
                                 'Годовая': 5}
        self.seasonality = self.seasonality_dict[data_frequency]
        self.original_timeseries = original_timeseries
        self.transformed_time_series = original_timeseries
        self.test_stationarity_code = None
        self.transformation_function = lambda x: x
        self.label = None
        self.d = 0
        self.D = 0

    def test_custom_difference(self, custom_transformation_size):
        self.d = custom_transformation_size[0]
        self.D = custom_transformation_size[1]

        self.transformed_time_series = self.original_timeseries.diff(self.d).diff(self.seasonality * self.D).dropna()
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')
        self.transformation_function = lambda x: x

        self.test_stationarity_code = '''
                # Applying Augmented Dickey-Fuller test
                dftest = adfuller(df.diff({}).diff({}).dropna(), autolag='AIC')
                '''.format(self.d, self.D)

        self.label = 'Пользовательские параметры' if self.dftest[0] < self.dftest[4]['1%'] else None

        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality
    
    def test_absolute_data(self):

        self.dftest = adfuller(self.original_timeseries, autolag='AIC')
        
        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                dftest = adfuller(df, autolag='AIC')
                    '''
        self.label = 'Без трансформации' if self.dftest[0] < self.dftest[4]['1%'] else None
        
        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality

    def test_first_difference(self):

        self.transformed_time_series = self.original_timeseries.diff().dropna()
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')

        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                dftest = adfuller(df.diff().dropna(), autolag='AIC')
                    '''
        self.label = 'Первое дифференцирование' if self.dftest[0] < self.dftest[4]['1%'] else None
        self.d = 1
        self.D = 0
        
        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality
        
    def test_log_transformation(self):

        self.transformed_time_series = np.log1p(self.original_timeseries)
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')
        self.transformation_function = np.log1p

        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                df = np.log1p(df) 
                dftest = adfuller(np.log1p(df), autolag='AIC')
                    '''
        self.label = 'Логарифмическое' if self.dftest[0] < self.dftest[4]['1%'] else None
        self.d = 0
        self.D = 0

        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality

    def test_seasonal_difference(self):

        self.transformed_time_series = self.original_timeseries.diff(self.seasonality).dropna()
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')
        self.transformation_function = lambda x: x

        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                dftest = adfuller(df.diff({}).dropna(), autolag='AIC')
                    '''.format(self.seasonality)
        self.label = 'Сезонное' if self.dftest[0] < self.dftest[4]['1%'] else None
        self.d = 0
        self.D = 1

        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality
    
    def test_log_difference(self):

        self.transformed_time_series = np.log1p(self.original_timeseries).diff().dropna()
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')
        self.transformation_function = np.log1p

        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                df = np.log1p(df)
                dftest = adfuller(df.diff().dropna(), autolag='AIC')
                    '''
        self.label = 'Первое логарифмическое' if self.dftest[0] < self.dftest[4]['1%'] else None
        self.d = 1
        self.D = 0

        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality
    
    def test_seasonal_log_difference(self):

        self.transformed_time_series = np.log1p(self.original_timeseries).diff().diff(self.seasonality).dropna()
        self.dftest = adfuller(self.transformed_time_series, autolag='AIC')
        self.transformation_function = np.log1p

        self.test_stationarity_code = '''
                # Приведение теста Дики-Фуллера
                df = np.log1p(df)
                dftest = adfuller(df.diff().diff({}).dropna(), autolag='AIC')
                '''.format(self.seasonality)

        self.label = 'Логарифмическое + Сезонное' if self.dftest[0] < self.dftest[4]['1%'] else None
        self.d = 1
        self.D = 1

        return self.dftest, self.transformed_time_series, self.label, self.d, self.D, self.transformation_function, \
               self.test_stationarity_code, self.seasonality

