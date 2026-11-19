import numpy as np
import jax.numpy as jnp
from scipy.stats import levy_stable

from tail_risk.core import calc_hill_curve, create_iqr_band, K_RATIO


def simulate_data(
    alphas: tuple[float] = (1.5, 1.7, 1.9), T: int = 5250, N: int = 50
) -> dict:
    k_max = int(K_RATIO * T)

    results = {}
    for alpha in alphas:
        simulated_data = levy_stable.rvs(alpha, beta=0, size=(T, N), random_state=0)

        losses = -jnp.asarray(simulated_data)
        hill_curve = calc_hill_curve(losses, k_max)
        hill_curve = np.asarray(hill_curve)
        # simulations[alpha] = hill_curve

        iqr_band = create_iqr_band(hill_curve, k_max, T)
        results[r"simulated $\alpha$ = " + f"{alpha}"] = iqr_band

    return results
