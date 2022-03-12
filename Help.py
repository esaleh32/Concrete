import streamlit as st
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from PIL import Image
image = Image.open('image.jpg')
def app():
     st.write('This app was programmed as part of my article, namely **"A Comprehensive Evaluation of Existing and New Model-identification approaches for Non-destructive Concrete Strength Assessment"**, submitted to **"Construction and Building materials"** jounral')
     st.write('It allows to estimate the minimum number of cores for reliable estimation of mean concrete strength and concrete variability by defining the five-inputs:')
     st.write('1)Quality of measurements')
     st.write('2)Accepted error in mean estimation')
     st.write('3)Accepted error in variability estimation')
     st.write('4)Accepted risk of wrong estimation')
     st.write('5)Coefficient of variation of NDT measurements')
     st.write('The concept of accepted error and risk in the estimation can be illustrated in following figure')
     st.image(image, caption='The concept of CDF used for evaluating the uncertainty (risk) in the estimation ')



