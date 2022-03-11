import streamlit as st
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
import scipy
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statistics
import pickle
def app():

    st.write("""
    # Concrete Strength Assessment  App

    **This app estimates the minimum number of cores for concrete strength assessment**
    """)



    #x=st.text_input("How could you elaborate NLP?")

    #st.number_input('Enter your age', 5, 100, 77)
    #st.number_input('Enter your age', 5.5)
    #st.write(df.iloc[:,1])

    st.sidebar.markdown("""
    Input variables for the determination of the minimum required number of cores
    """)
    def user_input_features():
        island = st.sidebar.selectbox('Quality of measurements',('High','Average','Low'))
        #sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Accepted error in the mean estimation %', 0,100,10)
        bill_depth_mm = st.sidebar.slider('Accepted error in the variability estimation %', 0,100,25)
        flipper_length_mm = st.sidebar.slider('Accepted risk %', 0,100,10)
        body_mass_g = st.sidebar.slider('coefficient of variation of rebound hammer readings, VR (units):', 0.0,1.0,0.4)


        data = {'qm': island,
                    'em': bill_length_mm,
                    'esd': bill_depth_mm,
                    'ar': flipper_length_mm,
                    'vr': body_mass_g

                    }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()



    t=input_df.qm=='Average'

    sq=input_df.qm=='Low'


    st.subheader('Minimum required number of cores for reliable mean estimation')
    #df=pd.read_csv('hea2.csv')
    #df.columns=['nc','risk','sd']
    #df.to_csv('hea2.csv')
    #X=df.iloc[:,df.columns!='nc']
    #y=df.iloc[:,df.columns=='nc']
    # Import the model we are using
    # Instantiate model with 1000 decision trees
    # Train the model on training data
    #MARS_model=Earth(max_terms=50,max_degree=1)
    #MARS_model_fitted=MARS_model.fit(X,y)
    rf = pickle.load(open('rf.pkl', 'rb'))

    ss=np.array([input_df.iloc[0,input_df.columns=='ar'],input_df.iloc[0,input_df.columns=='vr']*30])
    y_pred=rf.predict(ss.reshape(1,-1))+((10-np.float(input_df.iloc[0,input_df.columns=='em']))/10)
    y_pred=np.round(y_pred,0)
    if t.any():
        y_pred=y_pred+2
    if sq.any():
        y_pred=y_pred+3
    if y_pred<3:
        y_pred=3
    st.write(f"{int(y_pred)}")
    st.subheader('Minimum required number of cores for reliable variability estimation')
    rf2 = pickle.load(open('rf2.pkl', 'rb'))

    #df=pd.read_csv('hea6.csv')
    #X=df.iloc[:,df.columns!='nc']
    #y=df.iloc[:,df.columns=='nc']
    #MARS_model_fitted=MARS_model.fit(X,y)
    ss=np.array([input_df.iloc[0,input_df.columns=='ar'],input_df.iloc[0,input_df.columns=='vr']*25])

    y_pred=rf2.predict(ss.reshape(1,-1))+((25-np.float(input_df.iloc[0,input_df.columns=='esd']))/25)*3
    y_pred=np.round(y_pred,0)
    if t.any():
        y_pred=y_pred+2
    if sq.any():
        y_pred=y_pred+3
    if y_pred<3:
        y_pred=3
    st.write(f"{int(y_pred)}")
