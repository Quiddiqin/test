import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

def main():
    st.title("Regressione Immobili")
    newmodel = joblib.load('regression_test.pkl')
    newmodel

    x1 = st.slider('Inserisci crim', 0., 1000., 3.)
    x2 = st.slider('Inserisci zn', 0., 1000., 10.)
    x3 = st.slider('Inserisci indus', 0., 1000., 15.)
    x4 = st.slider('Inserisci chas', 0., 1000., 0.)
    x5 = st.slider('Inserisci nox', 0., 100., 1.)
    x6 = st.slider('Inserisci rm', 0., 100., 5.)
    x7 = st.slider('Inserisci age', 0., 100., 50.)
    x8 = st.slider('Inserisci dis', 0., 100., 2.)
    x9 = st.slider('Inserisci rad', 0., 100., 25.)
    x10 = st.slider('Inserisci tax', 0., 100., 20.)
    x11 = st.slider('Inserisci ptratio', 0., 100., 10.)
    x12 = st.slider('Inserisci b', 0., 1000., 525.)
    x13 = st.slider('Inserisci lsta', 0., 100., 25.)

    st.header(newmodel.predict([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]]))



    uploaded_file = st.file_uploader("Choose a file",type={"csv"})
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        X = df.iloc[:,-1]
        st.write(X)

        st.header('**Predizioni**')
        pred = newmodel.predict(X)
        st.write(newmodel.predict(X))

        df['Prediction'] = pred
        st.header('**Dataframe finale con le predizioni**')
        st.write(df)
        
        st.download_button(label= 'Download file finale', data=df.to_csv().encode('utf-8'), file_name='risultato.csv', mime='text/csv')


if __name__ == "__main__":
    main()


#streamlit run app.py