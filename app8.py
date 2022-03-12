import streamlit as st
from multiapp import MultiApp
import Home2
import ndf2
import cores
import Help
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

app = MultiApp()

# Add all your application here
#app.add_app("Home", Home.app)
app.add_app("Home", Home2.app)
app.add_app("Minimum number of cores", cores.app)
app.add_app("Conversion models", ndf2.app)
app.add_app("Help", Help.app)


# The main app
app.run()
