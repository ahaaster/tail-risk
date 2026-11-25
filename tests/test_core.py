import jax
import jax.numpy as jnp

from tail_risk.config import VAR_MODELS
from tail_risk.core import hill_curve_1d, kupiec_test, christoffersen_test
from tail_risk.core import NU_GRID, unit_variance_t_quantiles, log_likelihood_per_nu

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
    sigma_function = VAR_MODELS["normal_rolling"]

    logret = 0.01 * jax.random.normal(rng_key, (T, N))

    value_at_risk = sigma_function(logret, window=window, level=level)
    exceedances = -logret[window:] > value_at_risk

    forecast_count = exceedances.size
    binomial_std_error = jnp.sqrt(level * (1 - level) / forecast_count)

    assert abs(exceedances.mean() - level < 4 * binomial_std_error)


def test_normal_var_has_no_look_ahead():
    T, N = 1_000, 3
    window = 500
    rng_key = jax.random.key(2)
    sigma_function = VAR_MODELS["normal_rolling"]

    logret = 0.01 * jax.random.normal(rng_key, (T, N))
    # Manually create a crash event
    crash = logret.at[window].set(-1.0)

    var_clean = sigma_function(logret, window=window)
    var_crash = sigma_function(crash, window=window)

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


def test_ewma_var_has_no_look_ahead():
    T, N = 1_000, 3
    window = 504
    rng_key = jax.random.key(3)
    sigma_function = VAR_MODELS["normal_ewma"]

    logret = 0.01 * jax.random.normal(rng_key, (T, N))
    # Manually create a crash event
    crash = logret.at[window].set(-1.0)

    var_clean = sigma_function(logret, window=window)
    var_crash = sigma_function(crash, window=window)

    assert var_clean.shape == (T - window, N)
    assert jnp.allclose(
        var_clean[0], var_crash[0]
    ), "The rolling prediction for the crash day should not expect it"
    assert not jnp.allclose(
        var_clean[1], var_crash[1]
    ), "The day after the crash, predictions should be affected"


def test_christiffersen_fixed_value_and_edge_case():
    sequence = jnp.array([0, 1, 1, 0, 0, 1], dtype=bool)
    no_breach = jnp.zeros(6, dtype=bool)
    exceedances = jnp.stack([sequence, no_breach], axis=1)

    lr_stat, _ = christoffersen_test(exceedances)
    assert jnp.isclose(
        lr_stat[0], 0.1386, atol=1e-3
    ), "Precalculated value comparison test"
    assert jnp.isfinite(lr_stat[1]), "no exceedances at all should not result in NaN"


def test_fit_nu_recovers_student_t():
    T = 50_000
    true_nu = 5.0
    rng_key = jax.random.key(4)

    raw_t = jax.random.t(rng_key, true_nu, (T, 1))
    standardised_returns = raw_t * jnp.sqrt((true_nu - 2) / true_nu)

    nu_likelihoods = log_likelihood_per_nu(standardised_returns)
    nu_hat = NU_GRID[jnp.argmax(nu_likelihoods, axis=0)]
    assert abs(nu_hat[0] - true_nu) < 1.0


def test_student_t_var_shape_and_sign():
    T, N = 1_200, 3
    window = 504
    rng_key = jax.random.key(5)
    sigma_function = VAR_MODELS["t_rolling"]

    logret = 0.01 * jax.random.t(rng_key, 4.0, (T, N))
    value_at_risk = sigma_function(logret, window=window, refit_window=252)

    assert value_at_risk.shape == (T - window, N)
    assert jnp.all(value_at_risk > 0)


def test_unit_variance_t_quantile_values():
    level = 0.01
    quantiles = unit_variance_t_quantiles(level)

    nu_index = jnp.argmin(jnp.abs(NU_GRID - 4.0))

    assert jnp.isclose(quantiles[nu_index], -2.649, atol=1e-3)
    assert jnp.isclose(quantiles[-1], -2.37, atol=0.02)
