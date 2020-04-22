import numpy as np

def mean_abs_pct_error(actual_values, forecast_values):

    err=0
    for i in range(len(forecast_values)):
        err += np.abs(actual_values.values[i] - forecast_values.values[i]) / actual_values.values[i]
    return err * 100/len(forecast_values)
