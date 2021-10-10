import streamlit as st
import numpy as np
import pandas as pd
from gasNcity import GasNCity
import base64


def get_table_download_link(df_d):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df_d.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    return href


def set_sidebar():
    st.sidebar.title('Установить параметры')
    data = st.sidebar.file_uploader(label='Загрузите данные для предсказаний', type=['csv'])

    st.sidebar.markdown("Как альтернатива, можно ввести параметры вручную")
    st.sidebar.markdown("Установить расходы")
    Q_1 = st.sidebar.number_input('Q_1', 1.2, step=0.05)
    Q_2 = st.sidebar.number_input('Q_2', 0.6, step=0.05)
    Q_3 = st.sidebar.number_input('Q_3', 0.6, step=0.05)
    Q_4 = st.sidebar.number_input('Q_4', 0.6, step=0.05)
    Q_5 = st.sidebar.number_input('Q_5', 0.6, step=0.05)
    Q_6 = st.sidebar.number_input('Q_6', 3.0, step=0.05)
    Q_7 = st.sidebar.number_input('Q_7', 1.2, step=0.05)
    Q_8 = st.sidebar.number_input('Q_8', 1.8, step=0.05)

    QGRS_1 = st.sidebar.number_input('QGRS_1', 16., step=0.2)
    QGRS_2 = st.sidebar.number_input('QGRS_2', 16., step=0.2)

    QPlant_1 = st.sidebar.number_input('QPlant_1', 2.6, step=0.1)
    QPlant_2 = st.sidebar.number_input('QPlant_2', 3.8, step=0.1)
    QPlant_3 = st.sidebar.number_input('QPlant_3', 4.4, step=0.1)
    QPlant_4 = st.sidebar.number_input('QPlant_4', 5.2, step=0.1)

    st.sidebar.markdown("Установить давление")
    P_1 = st.sidebar.number_input('P_1', 220000, step=10000)
    P_2 = st.sidebar.number_input('P_2', 200000, step=10000)
    P_3 = st.sidebar.number_input('P_3', 210000, step=10000)
    P_4 = st.sidebar.number_input('P_4', 220000, step=10000)
    P_5 = st.sidebar.number_input('P_5', 220000, step=10000)
    P_6 = st.sidebar.number_input('P_6', 220000, step=10000)
    P_7 = st.sidebar.number_input('P_7', 260000, step=10000)
    P_8 = st.sidebar.number_input('P_8', 230000, step=10000)
    P_9 = st.sidebar.number_input('P_9', 280000, step=10000)

    PGRS_1 = st.sidebar.number_input('PGRS_1', 350000, step=10000)
    PGRS_2 = st.sidebar.number_input('PGRS_2', 350000, step=10000)

    st.sidebar.markdown("Здесь можно задать перекрытие вентилей")
    v1_zero = st.sidebar.checkbox("Перекрыт вентиль №1", value=False)
    v2_zero = st.sidebar.checkbox("Перекрыт вентиль №2", value=False)
    v3_zero = st.sidebar.checkbox("Перекрыт вентиль №3", value=False)
    v4_zero = st.sidebar.checkbox("Перекрыт вентиль №4", value=False)
    v5_zero = st.sidebar.checkbox("Перекрыт вентиль №5", value=False)
    v6_zero = st.sidebar.checkbox("Перекрыт вентиль №6", value=False)
    v7_zero = st.sidebar.checkbox("Перекрыт вентиль №7", value=False)
    v8_zero = st.sidebar.checkbox("Перекрыт вентиль №8", value=False)
    v9_zero = st.sidebar.checkbox("Перекрыт вентиль №9", value=False)
    v10_zero = st.sidebar.checkbox("Перекрыт вентиль №10", value=False)
    v11_zero = st.sidebar.checkbox("Перекрыт вентиль №11", value=False)
    v12_zero = st.sidebar.checkbox("Перекрыт вентиль №12", value=False)

    if data:
        st.markdown('read data')
        try:
            return pd.read_csv(data)
        except:
            st.markdown('second try')
            return pd.read_csv(data, sep=';')

    vars = locals()
    dct = {var: vars[var] for var in vars if var.startswith('P') or var.startswith('Q')}
    for i in range(1, 13):
        if vars[f'v{i}_zero']:
            dct[f'valve_{i}'] = 0

    return pd.DataFrame(dct, index=[0])


def run(df):

    st.title("Предложенные режимы управления")
    st.markdown("Идет расчет. Ожидаемое время расчета: 3.5 минуты на одну задачу")
    model = GasNCity(load_models=True, folder_path='models')
    res = model.find_valves(df)
    # res = df
    st.dataframe(res)
    st.markdown(get_table_download_link(res), unsafe_allow_html=True)


if __name__ == '__main__':
    good_cols = [f'Q_{i}' for i in range(1, 8)] + \
        [f'QPlant_{i}' for i in range(1, 5)] + \
        [f'P_{i}' for i in range(1, 10)] + \
        ['QGRS_1', 'QGRS_2', 'PGRS_1', 'PGRS_2']

    df = set_sidebar()
    run(df[good_cols])
    # if st.sidebar.button('Начать расчет'):
    #     run(df[good_cols])