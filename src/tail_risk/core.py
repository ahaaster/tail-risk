import jax
import numpy as np
import jax.numpy as jnp

K_RATIO = 0.10


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


def create_iqr_band(hill_curves, k_max: int, T: int) -> dict:
    quartiles = np.array([25, 50, 75])
    quartiles = np.percentile(hill_curves, quartiles, axis=0)

    # x-axis indexing to make different Ts per market consistent
    sample_share = np.arange(1, k_max + 1) / T

    return {"share": sample_share, "quartiles": quartiles}
