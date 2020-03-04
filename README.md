# GP Kernel Appl
This project illustrates how observations, and different kernels, effect GP inference and the resulting posterior curve/fit. 

For simplicity, I hear focus on Gaussian Likelihoods, so that inference is tractable and exactly solvable. In the future, the app will allow users to draw from GP priors with different kernels to see induced properties (e.g. differentiability) of the resulting posteriors.

The app can be run by navigating to the ```src/kedro_app/nodes/``` folder, and entering
```streamlit run streamlit_app.py```. A new tab will open in your system's default browser.