import streamlit as st
from multiapp import MultiApp
import Home2
import data_stats
import ndf2
import cores
import Help

app = MultiApp()

# Add all your application here
#app.add_app("Home", Home.app)
app.add_app("Home", Home2.app)
app.add_app("Minimum number of cores", cores.app)
app.add_app("Conversion models", ndf2.app)
app.add_app("Help", Help.app)


# The main app
app.run()