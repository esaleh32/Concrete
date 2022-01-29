
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
from pyearth import Earth
st.set_page_config(layout="wide")

path = st.text_input('Enter the CSV file path that contains the NDT and core test measurements')
if path:
    df = pd.read_csv(path)
    df
else:
    df=pd.read_csv("example input file.csv")

st.write (
'Press to download an example CSV input file')
df2= pd.read_csv("example input file.csv")
@st.cache
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
csv = convert_df(df2)

st.download_button(
   "Example CSV input file",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)



#x=st.text_input("How could you elaborate NLP?")

#st.number_input('Enter your age', 5, 100, 77)
#st.number_input('Enter your age', 5.5)
#st.write(df.iloc[:,1])
st.sidebar.header('User Input Variables')
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
    error = st.sidebar.selectbox('Error variance in NDT / Error variance in cores, \u03BB',('Known','Unknown'))
    if error=='Known':
        delta = st.sidebar.slider('error variance in x / error variance in y',0.0,2.0,1.0)
    else:
        delta=3
    data = {'qm': island,
                'em': bill_length_mm,
                'esd': bill_depth_mm,
                'ar': flipper_length_mm,
                'vr': body_mass_g,
                'error':error,
                'delta':delta
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
st.subheader('Conversion model for mean and variability estimation')

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
if np.float(input_df.iloc[0,input_df.columns=='delta'])==3:
    delta=1
else:
    delta=np.float(input_df.iloc[0,input_df.columns=='delta'])
slope,intercept=np.round(deming_regresion(df,'NDT','Core',delta),2)
st.write()
y=df.Core
X=df.NDT
#slope=np.round(statistics.stdev(y)/statistics.stdev(X),2)
#intercept=np.round(np.mean(y)-slope*np.mean(X),2)

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
ax.scatter(df.iloc[:,0], df.iloc[:,1],color='purple',  marker='o',s=50,alpha=1,edgecolor='k')
ax.set_xlabel('NDT measurement')
ax.set_ylabel('Core test measurement')
#ax.plot([np.min(df.iloc[:,0]),np.max(df.iloc[:,0])],[k+g*np.min(df.iloc[:,0]),k+g*np.max(df.iloc[:,0])],color='magenta',label='Conversion model for local strengths estimation')
ax.plot(_x, _y, label='Conversion model for local strengths estimation',color='magenta')
ax.plot([np.min(df.iloc[:,0]),np.max(df.iloc[:,0])],[intercept+slope*np.min(df.iloc[:,0]),intercept+slope*np.max(df.iloc[:,0])],color='blue',linestyle='--',label='Conversion model for mean and variability estimation')
ax.legend(fontsize=6)
st.pyplot(fig)



st.subheader('Minimum required number of cores for reliable mean estimation')
df=pd.read_csv('hea2.csv')
df.columns=['nc','risk','sd']
#df.to_csv('hea2.csv')
X=df.iloc[:,df.columns!='nc']
y=df.iloc[:,df.columns=='nc']
MARS_model=Earth(max_terms=50,max_degree=1)
MARS_model_fitted=MARS_model.fit(X,y)
ss=np.array([input_df.iloc[0,input_df.columns=='ar'],input_df.iloc[0,input_df.columns=='vr']*30])
y_pred=MARS_model_fitted.predict(ss.reshape(1,-1))+((10-np.float(input_df.iloc[0,input_df.columns=='em']))/10)*3
if y_pred<3:
    y_pred=3
st.write(f"{int(y_pred)}")
st.subheader('Minimum required number of cores for reliable variability estimation')
df=pd.read_csv('hea6.csv')
X=df.iloc[:,df.columns!='nc']
y=df.iloc[:,df.columns=='nc']
MARS_model=Earth(max_terms=50,max_degree=1)
MARS_model_fitted=MARS_model.fit(X,y)
ss=np.array([input_df.iloc[0,input_df.columns=='ar'],input_df.iloc[0,input_df.columns=='vr']*25])

y_pred=MARS_model_fitted.predict(ss.reshape(1,-1))+((25-np.float(input_df.iloc[0,input_df.columns=='esd']))/25)*3

if y_pred<3:
    y_pred=3
st.write(f"{int(y_pred)}")

