import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from gp_regression import  gp_conditional, visualize_gp, kernel_rbf_full, plot_gp_samples, gp_prior_draw

st.title('Exploration of Gaussian Processes')
st.text("""This app is meant to show how sampling can be done with GPs, 
and how inferences about posterior functions can differ based on kernel functions""")

f_true = lambda x: np.sin(x*(2*np.pi))
f_obs = lambda x: np.sin(x*(2*np.pi)) + 0.2*np.random.normal(size=x.shape)
obs_location_file = 'obs_locations.pkl'
@st.cache
def create_data(npoints = 200):
    return np.linspace(-1,1,200)

data = create_data()

# Now, let's sample, add to x_obs, then rerun and keep vizualizing.
# Once we iron out this part of the app, merge to master, then branch and
# make second "release"
# Eventually, define a drop down for different choices of kernel functions.
try:
    with open(obs_location_file, 'rb') as file_:
        obs_locations = pickle.load(file_)
except FileNotFoundError:
    obs_locations = []

x_new = st.number_input('Where should we measure? Please enter an x-coordinate value to take a measurement [-1,1]',
min_value=-1.,
max_value=1.,
value=0.,
step=0.01)
obs_locations.append(x_new)
st.text("""Let's see some prior draws""")
prior_draws = gp_prior_draw(kernel_func=kernel_rbf_full, domain = data, n_samples=50)
st.altair_chart(plot_gp_samples(data, prior_draws))
st.text("""Now, condition on observations (by default at 0)""")
# By default, number_input always returns default value of 0
# If I have data, let's condition and go

# Condition and Vizualize
alt_plt = visualize_gp(x_obs = np.array(obs_locations), x_coords=data, kernel_func=kernel_rbf_full, f_true=f_true,f_obs=f_obs)
st.altair_chart(alt_plt)
with open(obs_location_file, 'wb') as file_:
    if st.button('Clear Obs'):
        pickle.dump([], file=file_)    
    else:
        pickle.dump(obs_locations, file=file_)