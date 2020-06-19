import numpy as np

def m_error(actual_values, forecast_values):
    '''
   Функция вычисляет среднюю абсолютную ошибку в процентах
    '''
    err=0
    for i in range(len(forecast_values)):
        err += np.abs(actual_values.values[i] - forecast_values.values[i]) / actual_values.values[i]
    return err * 100/len(forecast_values)
