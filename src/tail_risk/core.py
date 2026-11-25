import jax
import numpy as np
import jax.numpy as jnp

from jax.scipy.stats import norm, chi2
from jax.scipy.special import xlogy


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


def rolling_normal_variance(
    logret, window: int = 504, level: float = 0.01, demean: bool = False
) -> jax.Array:
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

    z_level = norm.ppf(level)
    sqrt_var = jnp.sqrt(var_hat)
    return -(mean_hat + sqrt_var * z_level)


def ewma_normal_variance(
    logret, window: int = 504, level: float = 0.01, decay: float = 0.94
) -> jax.Array:
    logret = jnp.asarray(logret)
    initial_variance = jnp.mean(logret[:window] ** 2, axis=0)

    def update(var_previous, return_previous):
        var_next = decay * var_previous + (1 - decay) * return_previous**2
        return var_next, var_next

    _, later_vars = jax.lax.scan(update, initial_variance, logret[window:-1])
    var_hat = jnp.concatenate([initial_variance[None, :], later_vars])

    return -jnp.sqrt(var_hat) * norm.ppf(level)


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
