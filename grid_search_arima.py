import numpy as np
import statsmodels.api as sm
import streamlit as st

def grid_search_arima(train_data, exog,  p_range, q_range, P_range, Q_range, d=1, D=1, s=12):

    best_model_aic = np.Inf 
    best_model_bic = np.Inf 
    best_model_hqic = np.Inf
    best_model_order = (0, 0, 0)
    models = []
    with st.spinner('Поиск наилучших параметров. Ожидайте...'):
        for p_ in p_range:
            for q_ in q_range:
                for P_ in P_range:
                    for Q_ in Q_range:
                        try:
                            no_of_lower_metrics = 0
                            model = sm.tsa.statespace.SARIMAX(endog = train_data,
                                                                order = (p_, d, q_),
                                                                exog = exog,
                                                                seasonal_order = (P_, D, Q_, s),
                                                                enforce_invertibility=False).fit()
                            models.append(model)
                            if model.aic <= best_model_aic: no_of_lower_metrics += 1
                            if model.bic <= best_model_bic: no_of_lower_metrics += 1
                            if model.hqic <= best_model_hqic:no_of_lower_metrics += 1
                            if no_of_lower_metrics >= 2:
                                best_model_aic = np.round(model.aic,0)
                                best_model_bic = np.round(model.bic,0)
                                best_model_hqic = np.round(model.hqic,0)
                                best_model_order = (p_, d, q_, P_, D, Q_, s)
                                current_best_model = model
                                resid = np.round(np.expm1(current_best_model.resid).mean(), 3)
                                models.append(model)
                        except:
                            pass
    st.success('Поиск завершен!')
    st.markdown('')
    st.markdown('### Результаты лучшей модели')
    st.text(current_best_model.summary())
    return best_model_order
