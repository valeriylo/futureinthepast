import numpy as np
import statsmodels.api as sm
import streamlit as st

def train(Y, p, d, q, P, D, Q, s, exog_variables=None, quiet=False):
    '''
    Функция для обучения модели с помощью SARIMAX
    '''
    waiting_messages = ['Повторим законы робототехники: Первый закон: Робот не может...',
                        'На Саракш и обратно!',
                        'Ищем пропавшую звуковую отвертку...'
                        'Детям требуется куда больше времени!',
                        'Обжарить с каждой стороны по 2 минуты',
                        'Zzzzzzzzz...',
                        'Делу время, обучению видеокарта.',
                        'Запускаем машину времени...',
                        'Складываем, умножаем и логарифмируем...',
                        'It`s bigger on the inside!']

    mod = sm.tsa.statespace.SARIMAX(Y,
                                    order = (p, d, q),
                                    exog=exog_variables,
                                    seasonal_order = (P, D, Q, s),
                                    enforce_invertibility=False
                                    )
    if quiet:
        try:
            results = mod.fit()
        except np.linalg.LinAlgError:
            mod = sm.tsa.statespace.SARIMAX(Y,
                                    order = (p, d, q),
                                    exog=exog_variables,
                                    seasonal_order = (P, D, Q, s),
                                    enforce_invertibility=False,
                                    initialization='approximate_diffuse'
                                    )
            results = mod.fit()

    else:
        with st.spinner(np.random.choice(waiting_messages)):
            try:
                results = mod.fit()
            except np.linalg.LinAlgError:
                mod = sm.tsa.statespace.SARIMAX(Y,
                                        order = (p, d, q),
                                        exog=exog_variables,
                                        seasonal_order = (P, D, Q, s),
                                        enforce_invertibility=False,
                                        initialization='approximate_diffuse'
                                        )
                results = mod.fit()
        st.success('Выполнено!')
        
        try:
            st.text(results.summary())
        except:
            pass
    return results
