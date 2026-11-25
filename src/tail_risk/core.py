import jax
import numpy as np
import scipy.stats
import jax.numpy as jnp

from typing import Callable
from jax.scipy.stats import norm, chi2
from jax.scipy.special import xlogy, gammaln

# Degrees of freedom to test for fit
NU_GRID = jnp.arange(2.5, 30.05, 0.1)


def hill_curve_1d(losses: jax.Array, k_max: int) -> jax.Array:
    """calculates alpha(k) for k values [1, k_max]"""
    sorted_losses = jnp.sort(losses, descending=True)
    sorted_losses = sorted_losses[: k_max + 1]
    log_losses = jnp.log(sorted_losses)

    k_values = jnp.arange(1, k_max + 1)
    log_mean = jnp.cumsum(log_losses[:k_max]) / k_values

    inverse_alpha = log_mean - log_losses[1 : k_max + 1]
    return 1.0 / inverse_alpha


def calc_hill_curve(losses: jax.Array, k_max: int) -> jax.Array:
    losses = jnp.atleast_2d(losses)
    return jax.vmap(hill_curve_1d, in_axes=(1, None))(losses, k_max)


def calc_quartiles(values, axis: int = 0) -> np.ndarray:
    quartiles = np.array([25, 50, 75])
    return np.percentile(values, quartiles, axis=axis)


def create_iqr_band(hill_curves, k_max: int, T: int) -> dict:
    quartiles = calc_quartiles(hill_curves, axis=0)
    # x-axis indexing to make different Ts per market consistent
    sample_share = np.arange(1, k_max + 1) / T

    return {"share": sample_share, "quartiles": quartiles}


def rolling_sigma(logret, window: int = 504, demean: bool = False) -> jax.Array:
    logret = jnp.asarray(logret)
    T, N = logret.shape

    zero_row = jnp.zeros((1, N))
    cumsum_logret = jnp.concatenate([zero_row, jnp.cumsum(logret, axis=0)])
    cumsum_squared = jnp.concatenate([zero_row, jnp.cumsum(logret**2, axis=0)])

    window_sum = cumsum_logret[window:T] - cumsum_logret[: T - window]
    window_sum_squared = cumsum_squared[window:T] - cumsum_squared[: T - window]

    if demean:
        mean_hat = window_sum / window
        var_hat = (window_sum_squared - window * mean_hat**2) / (window - 1)
    else:
        mean_hat = jnp.zeros_like(window_sum)
        var_hat = window_sum_squared / window

    return jnp.sqrt(var_hat)


def ewma_sigma(logret, window: int = 504, decay: float = 0.94) -> jax.Array:
    logret = jnp.asarray(logret)
    initial_variance = jnp.mean(logret[:window] ** 2, axis=0)

    def update(var_previous, return_previous):
        var_next = decay * var_previous + (1 - decay) * return_previous**2
        return var_next, var_next

    _, later_vars = jax.lax.scan(update, initial_variance, logret[window:-1])
    var_hat = jnp.concatenate([initial_variance[None, :], later_vars])

    return jnp.sqrt(var_hat)


def normal_var(
    logret, sigma_function: Callable, window: int, level: float = 0.01
) -> jax.Array:
    sigma_hat = sigma_function(logret, window)
    return -sigma_hat * norm.ppf(level)


def kupiec_test(exceedances: jax.Array, T: int, level: float = 0.01):
    rate_hat = exceedances / T

    def bernoulli_log_likelihood(probability):
        return xlogy(T - exceedances, 1 - probability) + xlogy(exceedances, probability)

    lr_statistic = -2 * (
        bernoulli_log_likelihood(level) - bernoulli_log_likelihood(rate_hat)
    )
    p_value = chi2.sf(lr_statistic, df=1)

    return lr_statistic, p_value


def christoffersen_test(exceedances: jax.Array):
    assert (
        exceedances.dtype == jnp.bool_ and exceedances.ndim == 2
    ), "boolean (days, stocks) excpected"

    previous_day = exceedances[:-1]
    current_day = exceedances[1:]

    # n_{ij} where 1 means an exceedance, first index is t-1, second is t
    n00 = jnp.sum(~previous_day & ~current_day, axis=0)
    n01 = jnp.sum(~previous_day & current_day, axis=0)
    n10 = jnp.sum(previous_day & ~current_day, axis=0)
    n11 = jnp.sum(previous_day & current_day, axis=0)

    prob_01 = n01 / (n00 + n01)
    prob_11 = n11 / jnp.maximum(n10 + n11, 1)  # avoid dividing by 0
    null_probability = (n01 + n11) / (
        n00 + n01 + n10 + n11
    )  # null is independence of successive events

    markov_log_likelihood = (
        xlogy(n00, 1 - prob_01)
        + xlogy(n01, prob_01)
        + xlogy(n10, 1 - prob_11)
        + xlogy(n11, prob_11)
    )
    indep_log_likelihood = xlogy(n00 + n10, 1 - null_probability) + xlogy(
        n01 + n11, null_probability
    )

    lr_statistic = -2 * (indep_log_likelihood - markov_log_likelihood)
    p_value = chi2.sf(lr_statistic, df=1)
    return lr_statistic, p_value


def unit_variance_t_log_likelihood(returns: jax.Array, nu: float) -> jax.Array:
    const_part = 0.5 * jnp.log(jnp.pi * (nu - 2))
    constant = gammaln((nu + 1) / 2) - gammaln(nu / 2) - const_part

    log_term = jnp.log1p(returns**2 / (nu - 2))
    log_kernel = -(nu + 1) / 2 * log_term
    return jnp.sum(constant + log_kernel, axis=0)


@jax.jit
def log_likelihood_per_nu(
    standardised_returns: jax.Array, nu_grid: jax.Array = NU_GRID
) -> jax.Array:
    return jax.lax.map(
        lambda nu: unit_variance_t_log_likelihood(standardised_returns, nu), nu_grid
    )


def unit_variance_t_quantiles(level: float, nu_grid: jax.Array = NU_GRID) -> jax.Array:
    nu_values = np.asarray(nu_grid)
    factor = np.sqrt((nu_values - 2) / nu_values)
    # JAX implementation of scipy.stats has no ppf available
    quantiles = scipy.stats.t.ppf(level, nu_values) * factor

    return jnp.asarray(quantiles)


def fit_nu_per_block(logret, sigma_function, window: int, refit_window: int = 252):
    logret = jnp.asarray(logret)
    sigma_hat = sigma_function(logret, window)

    standardised_returns = logret[window:] / sigma_hat
    warmup_standardised_ret = logret[:window] / jnp.std(logret[:window], axis=0)

    running_log_likelihood = log_likelihood_per_nu(warmup_standardised_ret)
    n_forecasts = sigma_hat.shape[0]
    nu_index_per_block = []
    for block_start in range(0, n_forecasts, refit_window):
        # past_data = standardised_returns[:block_start]
        # fit_data = jnp.concat([warmup_standardised_ret, past_data])
        # nu_index_per_block.append(fit_to_nu_grid(fit_data))

        nu_index_per_block.append(jnp.argmax(running_log_likelihood, axis=0))
        new_block = standardised_returns[block_start : block_start + refit_window]
        running_log_likelihood += log_likelihood_per_nu(new_block)

    return sigma_hat, jnp.stack(nu_index_per_block)


def student_t_var(
    logret, sigma_function, window: int, level: float = 0.01, refit_window: int = 252
) -> jax.Array:
    sigma_hat, nu_index_per_block = fit_nu_per_block(
        logret, sigma_function, window, refit_window
    )
    quantile_arr = unit_variance_t_quantiles(level)

    value_at_risk_blocks = []
    for block_number, nu_index in enumerate(nu_index_per_block):
        block_start = block_number * refit_window
        sigma_block = sigma_hat[block_start : block_start + refit_window]
        value_at_risk_blocks.append(-sigma_block * quantile_arr[nu_index])

    return jnp.concat(value_at_risk_blocks)
