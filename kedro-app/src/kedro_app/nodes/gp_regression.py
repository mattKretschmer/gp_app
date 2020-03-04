import scipy
import numpy as np
import pandas as pd
import altair as alt

def gp_conditional(x_new:np.array, 
x:np.array, y:np.array, cov_function:callable, sigma_obs:float = 0.3):
""" Returns the predictive mean and variance at some new locations

Parameters
----------
x_new : np.array
    [description]
x : np.array
    [description]
y : np.array
    [description]
cov_function : callable
    [description]
sigma_obs : float, optional
    [description], by default 0.3

Returns
-------
[type]
    [description]
"""

    K = cov_function(x, x)
    K += sigma_obs*np.eye(N=K.shape[0])
    L = scipy.linalg.cholesky(K, lower=True)
    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y, lower=True))
    k_star = cov_function(x, x_new)
    f = k_star.T@alpha

    v = scipy.linalg.solve_triangular(L, k_star, lower=True)
    f_var = cov_function(x_new, x_new) - v.T@v
    n = x.shape[0]
    lml = -0.5*y.T@alpha - n*np.log(2*np.pi)/2 - np.sum(np.log(np.diag(L)))
    return f, f_var, lml


# Let's use numpy magic to make the full covariance matrix at once!
def kernel_rbf_full(x: np.array, y:np.array, length_scale:float=0.3):
    """  An implementation of the radial basis function gp covariance function.
    
    Parameters
    ----------
    x : np.array
        [description]
    y : np.array
        [description]
    length_scale : float, optional
        [description], by default 0.3
    
    Returns
    -------
    [type]
        [description]
    """
    assert isinstance(x, np.ndarray), "Inputs should be nd arrays"
    return np.exp(-np.power(np.subtract.outer(x,y)/length_scale,2))

def gp_prior_draw(kernel_func: callable, domain:np.array, n_samples = 25):
    """ Return n draws from the mean 0 GP prior with covariance defined by the kernel_func
    
    Parameters
    ----------
    kernel_func : callable
        [description]
    domain : np.array
        [description]
    n_samples : int, optional
        [description], by default 25
    
    Returns
    -------
    [type]
        [description]
    """
    n_points = domain.shape[0]
    return scipy.stats.multivariate_normal(mean = np.zeros(n_points), 
                                                cov = kernel_func(domain, domain),
                                               allow_singular=True).rvs(size=n_samples)

def plot_gp_samples(x:np.array, numpy_samples:np.array, max_samples:int = 50):
    """ Plot the samples/draws from a GP, calculated using vanilla numpy and scipy.
    These samples can be either from the conditional distribution (given obs) or from
    the GP prior.
    
    Parameters
    ----------
    x : np.array
        [description]
    numpy_samples : np.array
        [description]
    max_samples : int, optional
        [description], by default 50
    
    Returns
    -------
    [altair.vegalite.v4.api.Chart]
        Returns an altair Chart showing GP samples
    """
    samples, points = numpy_samples.shape
    samples_to_plot = numpy_samples
    idx = np.random.choice(np.arange(samples), size = np.minimum(samples, max_samples), replace=False)
    raw_samples = pd.DataFrame(samples_to_plot[idx,:].T)
    raw_samples.columns = ['sample_%d'%col for col in raw_samples.columns]
    raw_samples['x'] = x
    samples_plot = pd.melt(raw_samples, id_vars='x', var_name='sample_id')
    samples_plot.rename(columns={'value':'f(x)'}, inplace=True)
    samples_plot['label'] = 'sample'
    
    gp_sample_plot = alt.Chart(samples_plot).mark_line().encode(x='x', y='f(x)', detail='sample_id',
                                                              color=alt.Color('label', title='Legend'))
    return gp_sample_plot