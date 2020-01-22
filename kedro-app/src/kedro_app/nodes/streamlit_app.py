import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.title('Exploration of Gaussian Processes')

DATE_COLUMN = 'date/time'
f_true = lambda x: np.sin(x*(2*np.pi))
f_obs = lambda x: f_true(x) + 0.2*np.random.normal(size=x.shape)

@st.cache
def create_data(npoints = 200):
    return np.linspace(-1,1,200)

data = create_data()

if st.checkbox('Show noiseless function'):
    st.subheader('True Function')
    sns.lineplot(x=data, y=f_true(data))
    st.pyplot()
x_new = st.number_input('Where should we measure?')
# st.subheader('Number of pickups by hour:')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00'% hour_to_filter)
# st.map(filtered_data)