import scipy
import numpy as np
import pandas as pd
import altair as alt


def gp_conditional(
    x_new: np.array,
    x: np.array,
    y: np.array,
    cov_function: callable,
    sigma_obs: float = 0.3,
):
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
        Variance of observation noise, by default 0.3
    
    Returns
    -------
    [type]
        [description]
    """

    K = cov_function(x, x)
    K += sigma_obs * np.eye(N=K.shape[0])
    L = scipy.linalg.cholesky(K, lower=True)
    alpha = scipy.linalg.solve_triangular(
        L.T, scipy.linalg.solve_triangular(L, y, lower=True)
    )
    k_star = cov_function(x, x_new)
    f = k_star.T @ alpha

    v = scipy.linalg.solve_triangular(L, k_star, lower=True)
    f_var = cov_function(x_new, x_new) - v.T @ v
    n = x.shape[0]
    lml = -0.5 * y.T @ alpha - n * np.log(2 * np.pi) / 2 - np.sum(np.log(np.diag(L)))
    return f, f_var, lml


def kernel_rbf_full(x: np.array, y: np.array, length_scale: float = 0.3):
    """  An implementation of the radial basis function gp covariance function.
    Uses pure numpy!
    
    Parameters
    ----------
    x : np.array
        [description]
    y : np.array
        [description]
    length_scale : float, optional
        Length scale of RBF kernel, by default 0.3
    
    Returns
    -------
    [type]
        [description]
    """
    assert isinstance(x, np.ndarray), "Inputs should be nd arrays"
    return np.exp(-np.power(np.subtract.outer(x, y) / length_scale, 2))


def gp_prior_draw(kernel_func: callable, domain: np.array, n_samples=25):
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
    return scipy.stats.multivariate_normal(
        mean=np.zeros(n_points), cov=kernel_func(domain, domain), allow_singular=True
    ).rvs(size=n_samples)


def plot_gp_samples(x: np.array, numpy_samples: np.array, max_samples: int = 50):
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
    samples, _ = numpy_samples.shape
    samples_to_plot = numpy_samples
    idx = np.random.choice(
        np.arange(samples), size=np.minimum(samples, max_samples), replace=False
    )
    raw_samples = pd.DataFrame(samples_to_plot[idx, :].T)
    raw_samples.columns = ["sample_%d" % col for col in raw_samples.columns]
    raw_samples["x"] = x
    samples_plot = pd.melt(raw_samples, id_vars="x", var_name="sample_id")
    samples_plot.rename(columns={"value": "f(x)"}, inplace=True)
    samples_plot["label"] = "sample"

    gp_sample_plot = (
        alt.Chart(samples_plot)
        .mark_line()
        .encode(
            x="x",
            y="f(x)",
            detail="sample_id",
            color=alt.Color("label", title="Legend"),
        )
    )
    return gp_sample_plot


def visualize_gp(
    x_obs: np.array,
    x_coords: np.array,
    kernel_func: callable,
    f_true: callable,
    f_obs: callable,
):
    """ Given a grid we're defining each point on, and some other functions, 
    let's plot locations we're sampling from, then the true curve, samples, and the posterior.
    
    Parameters
    ----------
    x_obs : np.array
        [description]
    x_coords : np.array
        [description]
    kernel_func : callable
        [description]
    f_true : callable
        [description]
    f_obs : callable
        [description]
    """
    y_obs = f_obs(x_obs)
    y_true = f_true(x_coords)
    f_star, f_sigma_star, _ = gp_conditional(
        x_coords, x_obs, y_obs, kernel_func, sigma_obs=0.05
    )
    conditional_samples = scipy.stats.multivariate_normal(
        mean=f_star, cov=f_sigma_star, allow_singular=True
    ).rvs(size=200)
    conditional_samples_plot = plot_gp_samples(
        x_coords, conditional_samples
    ).mark_line(opacity=0.2)
    domain_ = ["observed", "truth", "posterior", "sample"]
    range_ = ["red", "black", "green", "blue"]
    observed_plot = (
        alt.Chart(pd.DataFrame({"x": x_obs, "f(x)": y_obs, "label": "observed"}))
        .mark_circle(size=100)
        .encode(
            x="x",
            y="f(x)",
            color=alt.Color("label", scale=alt.Scale(domain=domain_, range=range_)),
        )
    )
    true_chart = (
        alt.Chart(pd.DataFrame({"x": x_coords, "f(x)": y_true, "label": "truth"}))
        .mark_line()
        .encode(
            x="x",
            y="f(x)",
            color=alt.Color("label", scale=alt.Scale(domain=domain_, range=range_)),
        )
    )
    posterior_chart = (
        alt.Chart(pd.DataFrame({"x": x_coords, "f(x)": f_star, "label": "posterior"}))
        .mark_line()
        .encode(
            x="x",
            y="f(x)",
            color=alt.Color("label", scale=alt.Scale(domain=domain_, range=range_)),
        )
    )
    return alt.layer(
        conditional_samples_plot, observed_plot, true_chart, posterior_chart
    )
