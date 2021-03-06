import streamlit as st
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
from PIL import Image
image = Image.open('image.jpg')
image4 = Image.open('image4.jpg')
image2 = Image.open('image2.jpg')
image3 = Image.open('image3.jpg')



def app():
     st.title('Help page')
     st.write('These apps were programmed as part of my article, namely **"A Comprehensive Evaluation of Existing and New Model-identification approaches for Non-destructive Concrete Strength Assessment"**, submitted to **"Construction and Building materials"** jounral')

     st.subheader('Minimum number of cores app')
     st.write('It allows to estimate the minimum number of cores for reliable estimation of mean concrete strength and concrete variability by defining the five-inputs:')
     st.write('1)Quality of measurements')
     st.write('2)Accepted error in mean estimation')
     st.write('3)Accepted error in variability estimation')
     st.write('4)Accepted risk of wrong estimation')
     st.write('5)Coefficient of variation of NDT measurements')
     st.write('The concept of accepted error and risk in the estimation can be illustrated in following figure')
     st.image(image, caption='The concept of CDF used for evaluating the uncertainty (risk) in the estimation ')
     st.write ('It should be noted that the minimum numbers of cores that corrosponds to the above inputs were dervied using sythetic data analysis of NDT and core test measurements')
     st.write ('For example,the user may need to know the minimum number of cores for mean and variability estimation of concrete strength that corrosponds to:')
     st.write('High quality of measurements')
     st.write ('Accepted error in the mean estimation % =10%')
     st.write('Accepted error in the variability estimation % =25%')
     st.write('Accepted risk %=10%')
     st.write('coefficient of variation in NDT measurements= 0.4')
     st.write('choosing these inputs the minimum number of cores will be shown as below')
     st.image(image3)
     st.subheader('Conversion models app')
     st.write('The common practice of concrete strength assessment is to combine non-destructive techniques (NDT) with core test measurements to develop a conversion model that is used to estimate the strengths at NDT test locations. The figure below illustrates a graphical description of this assessment method.')
     st.image(image2, caption='Concrete strength assessment process using the combined results of NDT and core tests.')
     st.write ('The conversion model is a relationship that is used to convert the NDT test results spread out over the entire structure to in-situ compressive strength values.')
     st.write ('Based on an extensive analysis of sythetic data of NDT and core test measurements, a hybrid conversion model identification system that uses **the bi-objective approach** proposed by Alwash et al. for mean and variability estimation of concrete strength and **quantile regression** for local concrete strengths estimation is proposed.') 
     st.write ('Thus, this app estimates two conversion models one for the evaluation of mean and variability of concrete strength (from the bi-objective approach) and one for the estimation of local strengths (from quantile regression')
     st.write ('The user require to upload a file that contains the NDT and core test measurements obtained through the assessement process then it will provide the user with the necessary conversion models for concrete assessement, in addition, the app will provide visulazation of the data and the resulted conversion models as shown in the figure below')
     
     st.image(image4)
     st.subheader('details on the derivation of the minimum number of cores process')
     st.write ('  1) large sets of synthetic data with different properties were simulated')
     st.write('  2)  For each set (defined by mean concrete strength, concrete variability, quality of measurements, and VR that can be derived from the aforementioned properties), the CDF curves for different numbers of cores are constructed.')
     st.write('  3)  For a certain margin of error, the risk values were estimated from the CDF shown in the figure above. This will generate relationships between the number of cores and risk values for each set of synthetic data')
     st.write ('  4) From these relationships, the number of cores corresponding to a certain risk value is obtained')
     st.write ('It should be noted that the mimimum number of cores app does not apply to a very low mean concrete strength (less than 15 MPa; this strength class is common for non-structural concrete). As it was observed that the risk values corresponding to this strength class are very high; thus, require a lot of cores for reliable estimation of strength even larger than what is specified in the standards.') 


  
 








 
  




 


  
 






