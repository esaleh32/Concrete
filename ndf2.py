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

def app ():

    st.write("""
    # Concrete Strength Assessment  App

    **This app estimates the conversion models for concrete strength assessment **
    """)

    path=st.file_uploader("Upload CSV file that contains the NDT and core test measurements",type=['csv'])
    if path:
        df =pd.read_csv(path)
        df
    else:
        df=pd.read_csv("example input file.csv")

    st.write (
    'Press to download an example CSV input file')
    
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df)


    st.download_button(
           "Example CSV input file",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )

  

    def deming_regresion(df, X, y, delta = 1):

        cov = df.cov()
        mean_x = df[X].mean()
        mean_y = df[y].mean()
        s_xx = cov[X][X]
        s_yy = cov[y][y]
        s_xy = cov[X][y]

        slope = (s_yy - delta * s_xx + np.sqrt((s_yy - delta * s_xx) ** 2 + 4 * delta * s_xy ** 2)) / (2 * s_xy)

        intercept = mean_y - slope * mean_x

        return slope, intercept
    df.columns=['NDT','Core']
    #delta=statistics.variance(df.NDT)/statistics.variance(df.Core)


    delta=1
    slope,intercept=np.round(deming_regresion(df,'NDT','Core',delta),2)
    st.write()
    y=df.Core
    X=df.NDT
    #slope=np.round(statistics.stdev(y)/statistics.stdev(X),2)
    #intercept=np.round(np.mean(y)-slope*np.mean(X),2)
    st.subheader('Conversion model for mean concrete strength estimation')
    st.write(f"Core measurement = {intercept}+{slope}(NDT)", unsafe_allow_html=True)

    st.subheader('Conversion model for local strengths estimation')
    df.columns=['NDT','Core']
    model = smf.quantreg('Core ~ NDT',df)
    quantiles=[0.5]
    fits = [model.fit(q=q) for q in quantiles]
    _x = np.linspace(X.min(),X.max())
    for index, quantile in enumerate(quantiles):
        _y = fits[index].params['NDT'] * _x + fits[index].params['Intercept']

    #X = np.vstack([x, np.ones(len(x))]).T
    #lrr.fit(X, y)
    #g=np.round(lrr.coef_[0],2)
    #k=np.round(lrr.intercept_,2)
    g=np.round(fits[index].params['NDT'],2)
    k=np.round(fits[index].params['Intercept'],2)
    st.write(f"Core measurement = {k}+{g}(NDT)", unsafe_allow_html=True)

    st.subheader('Visualization')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(df.iloc[:,0], df.iloc[:,1],color='blue',  marker='o',s=50,alpha=1,edgecolor='k')
    ax.set_xlabel('NDT measurement')
    ax.set_ylabel('Core test measurement')
    #ax.plot([np.min(df.iloc[:,0]),np.max(df.iloc[:,0])],[k+g*np.min(df.iloc[:,0]),k+g*np.max(df.iloc[:,0])],color='magenta',label='Conversion model for local strengths estimation')
    ax.plot(_x, _y, label='Conversion model for local strengths estimation',color='red')
    ax.plot([np.min(df.iloc[:,0]),np.max(df.iloc[:,0])],[intercept+slope*np.min(df.iloc[:,0]),intercept+slope*np.max(df.iloc[:,0])],color='green',linestyle='--',label='Conversion model for mean and variability estimation')
    ax.legend(fontsize=6)
    st.pyplot(fig)
