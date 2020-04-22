import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import streamlit as st

def find_acf_pacf(timeseries, seasonality):

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    p_terms = 0
    q_terms = 0
    P_terms = 0
    Q_terms = 0

    lower_conf_int = -1.96/np.sqrt(len(timeseries.dropna()))
    upper_conf_int = 1.96/np.sqrt(len(timeseries.dropna()))

    pacf_values = sm.tsa.stattools.pacf(timeseries.dropna(), nlags = seasonality * 2, method='ywunbiased')

    acf_values = sm.tsa.stattools.acf(timeseries.dropna(), nlags = seasonality * 2, fft=False, unbiased=False)


    for value in pacf_values[1:]:
        if value >= upper_conf_int or value <= lower_conf_int:
            p_terms += 1
        else:
            break

    for value in acf_values[1:]:
        if value >= upper_conf_int or value <= lower_conf_int:
            q_terms += 1
        else:
            break

    if pacf_values[seasonality] >= upper_conf_int or pacf_values[seasonality] <= lower_conf_int:
        P_terms += 1
        if pacf_values[seasonality*2] >= upper_conf_int or pacf_values[seasonality*2] <= lower_conf_int:
            P_terms += 1

    if acf_values[seasonality] >= upper_conf_int or acf_values[seasonality] <= lower_conf_int:
        Q_terms += 1
        if acf_values[seasonality*2] >= upper_conf_int or acf_values[seasonality*2] <= lower_conf_int:
            Q_terms += 1

    sm.graphics.tsa.plot_acf(timeseries.dropna(), lags = seasonality * 2, ax=ax1, color='green')

    sm.graphics.tsa.plot_pacf(timeseries.dropna(), lags = seasonality * 2, ax=ax2, color='green')
    
    plt.subplots_adjust(hspace=.4)
    st.pyplot()

    return p_terms, q_terms, P_terms, Q_terms
