import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_ts(ts):
    '''
    Функция производит декомпозицию временного ряда
    '''
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    try:
        decomposition = seasonal_decompose(ts)

    except AttributeError:
        error_message = '''
                        Похоже стобец "Дата" в некорректном формате. Убедитесь, что он соответствует формату функции
                        Pandas to_datetime.
                        '''
        raise AttributeError(error_message)

    decomposition.seasonal.plot(color='green', ax=ax1, title='Сезонность')
    plt.legend('')

    decomposition.trend.plot(color='green', ax=ax2, title='Тренд')
    plt.legend('')

    decomposition.resid.plot(color='green', ax=ax3, title='Остатки')
    plt.legend('')
    plt.subplots_adjust(hspace=1)
    st.pyplot()
