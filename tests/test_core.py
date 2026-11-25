import jax
import jax.numpy as jnp

from tail_risk.core import hill_curve_1d, rolling_normal_variance, ewma_normal_variance
from tail_risk.core import kupiec_test, christoffersen_test

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


def test_normal_var_coverage_on_normal():
    T, N = 10_500, 20
    window, level = 500, 0.01
    rng_key = jax.random.key(1)

    logret = 0.01 * jax.random.normal(rng_key, (T, N))

    value_at_risk = rolling_normal_variance(logret, window, level)
    exceedances = -logret[window:] > value_at_risk

    forecast_count = exceedances.size
    binomial_std_error = jnp.sqrt(level * (1 - level) / forecast_count)

    assert abs(exceedances.mean() - level < 4 * binomial_std_error)


def test_normal_var_has_no_look_ahead():
    T, N = 1_000, 3
    window = 500
    rng_key = jax.random.key(2)

    logret = 0.01 * jax.random.normal(rng_key, (T, N))
    # Manually create a crash event
    crash = logret.at[window].set(-1.0)

    var_clean = rolling_normal_variance(logret, window)
    var_crash = rolling_normal_variance(crash, window)

    # The rolling prediction for the crash day should not expect it
    assert jnp.allclose(var_clean[0], var_crash[0])
    # The day after the crash, predictions should be affected
    assert not jnp.allclose(var_clean[1], var_crash[1])


def test_kupiec_hand_value_and_edge_case():
    T = 1_000
    test_arr = jnp.array([20, 10, 0])

    lr_stat, p_val = kupiec_test(test_arr, T, level=0.01)

    # Precalculated value comparison
    assert jnp.isclose(lr_stat[0], 7.827, atol=1e-3)
    # Obserbed rate is nominal
    assert jnp.isclose(lr_stat[1], 0.0)
    # Make sure NaN is not produced for x = 0
    assert jnp.isfinite(lr_stat[2])
    # Chi2 1% critical value test
    assert p_val[0] < 0.01


def test_ewma_variance_has_no_look_ahead():
    T, N = 1_000, 3
    window = 504
    rng_key = jax.random.key(3)

    logret = 0.01 * jax.random.normal(rng_key, (T, N))
    # Manually create a crash event
    crash = logret.at[window].set(-1.0)

    var_clean = ewma_normal_variance(logret, window)
    var_crash = ewma_normal_variance(crash, window)

    assert var_clean.shape == (T - window, N)
    # The rolling prediction for the crash day should not expect it
    assert jnp.allclose(var_clean[0], var_crash[0])
    # The day after the crash, predictions should be affected
    assert not jnp.allclose(var_clean[1], var_crash[1])


def test_christiffersen_fixed_value_and_edge_case():
    sequence = jnp.array([0, 1, 1, 0, 0, 1], dtype=bool)
    no_breach = jnp.zeros(6, dtype=bool)
    exceedances = jnp.stack([sequence, no_breach], axis=1)

    lr_stat, _ = christoffersen_test(exceedances)
    # Precalculated value comparison test
    assert jnp.isclose(lr_stat[0], 0.1386, atol=1e-3)
    # Make sure no exceedances at all do not result in NaN
    assert jnp.isfinite(lr_stat[1])
