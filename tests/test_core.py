import jax
import jax.numpy as jnp

from tail_risk.core import hill_curve_1d

jax.config.update("jax_enable_x64", True)


def test_hill_with_pareto_alpha():
    rng_key = jax.random.key(0)
    alpha = 3.0
    T = 100_000
    k = 2_000

    uni_distribution = jax.random.uniform(rng_key, (T,))
    pareto_losses = uni_distribution ** (-1.0 / alpha)
    alpha_estimate = hill_curve_1d(pareto_losses, k_max=k)[k - 1]

    standard_error = alpha / jnp.sqrt(k)
    assert abs(alpha_estimate - alpha) < 4 * standard_error
