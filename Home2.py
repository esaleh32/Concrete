import streamlit as st
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


def app():
    st.write("""
        # Concrete Strength Assessment  App

        **This app estimates the conversion models and minimum number of cores for concrete strength assessment **
        """)

    st.write('Navigate to **Minimum number of cores** page to estimate the minimum number of cores for concrete strength estimation')
    st.write ('Navigate to **Conversion models** page to estimate the necessary conversion models for concrete strength estimation')
    st.write ('Navigate to Help to display the application help system.')
