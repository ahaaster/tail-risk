import jax
import matplotlib

matplotlib.use("Agg")  # only render to file
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable

from tail_risk.config import K_RATIO, K_OVER_T_SHARES, MARKET_LABELS
from tail_risk.config import VAR_MODELS, VAR_WINDOW, VAR_LEVEL, MODEL_LABELS
from tail_risk.core import calc_hill_curve, create_iqr_band, calc_quartiles, kupiec_test
from tail_risk.simulation import simulate_data
from tail_risk.plots import plot_hill_curves, plot_ecdf, plot_exceedance_over_time

jax.config.update("jax_enable_x64", True)

WORKING_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WORKING_DIR / "data"
FIG_PATH = WORKING_DIR / "figures"
STOCK_SELECTION = ["AAPL", "C", "KO", "ASML.AS", "BNP.PA"]


def calculate_hill(log_returns: pd.DataFrame) -> (jnp.Array, int, int):
    # We invert the returns to ultimately sort the largest loss as first index
    losses = -jnp.asarray(log_returns.to_numpy())

    T, _ = losses.shape
    k_max = int(K_RATIO * T)

    hill_curves = calc_hill_curve(losses, k_max)
    return hill_curves, T, k_max


def select_stocks(
    log_returns: pd.DataFrame, hill_curves: jnp.Array, sample_share: jnp.Array
) -> list[tuple]:
    relevant_stocks = log_returns.columns.intersection(STOCK_SELECTION)

    list_of_stock_results = []
    for stock in relevant_stocks:
        idx = log_returns.columns.get_loc(stock)
        hill_curve = hill_curves[idx]

        # if stock == "C":
        #     sorted_returns = log_returns[stock]
        #     sorted_returns = sorted_returns.sort_values()
        #     print(sorted_returns.head(10))

        stock_results = (stock, sample_share, hill_curve)
        list_of_stock_results.append(stock_results)

    return list_of_stock_results


def run_var_backtest(
    log_returns: pd.DataFrame,
    market_name: str,
    model_name: str,
    model_function: Callable,
    window: int = VAR_WINDOW,
    level: float = VAR_LEVEL,
):
    losses = -jnp.asarray(log_returns.to_numpy())
    losses = losses[window:]

    value_at_risk = model_function(log_returns, window, level)
    exceedances = losses > value_at_risk

    forecast_count = exceedances.shape[0]
    exceed_count = exceedances.sum(axis=0)
    _, p_val = kupiec_test(exceed_count, forecast_count, level)

    exceed_rate = np.asarray(exceed_count / forecast_count)
    quarter, median, three_quart = calc_quartiles(exceed_rate)

    rejections = (p_val < 0.05).mean()

    results = {
        "market": MARKET_LABELS[market_name],
        "model": MODEL_LABELS[model_name],
        "median rate exceeded": f"{median:.2%}",
        "IQR-band": f"{quarter:.2%}-{three_quart:.2%}",
        "share rejected (5%)": f"{rejections:.1%}",
    }

    dates = log_returns.index[window:]
    exceedance_df = pd.DataFrame(np.asarray(exceedances), index=dates)
    return results, exceed_rate, exceedance_df


if __name__ == "__main__":
    DATA_PATH.mkdir(exist_ok=True)
    FIG_PATH.mkdir(exist_ok=True)
    data_paths = DATA_PATH.glob("logret_*.csv")

    market_results, ecdf_results, var_daily_shares = {}, {}, {}
    single_stocks, summary_rows, var_rows = [], [], []
    for data_path in data_paths:
        file_name = data_path.stem
        market_name = file_name.removeprefix("logret_").removesuffix("_daily")

        log_returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
        hill_curves, T, k_max = calculate_hill(log_returns)
        hill_curves = np.asarray(hill_curves)

        iqr_band = create_iqr_band(hill_curves, k_max, T)
        market_results[market_name] = iqr_band

        stock_results = select_stocks(log_returns, hill_curves, iqr_band["share"])
        single_stocks.extend(stock_results)

        ecdf_results[market_name] = {}
        for pct in K_OVER_T_SHARES:
            k = int(pct * T)
            alpha_hats = hill_curves[:, k - 1]
            ecdf_results[market_name][pct] = alpha_hats

            share_below_2_alpha = (alpha_hats < 2).mean()
            summary_rows.append(
                {
                    "market": market_name,
                    "k/T": f"{pct:.1%}",
                    "k": k,
                    r"median $\hat{\alpha}$": round(float(np.median(alpha_hats)), 2),
                    r"share $\hat{\alpha} < 2$": f"{share_below_2_alpha:.1%}",
                }
            )

        # Value-at-Risk section per market per model
        var_daily_shares[market_name] = {}
        for model_name, model_function in VAR_MODELS.items():
            var_row, var_rates, var_exceed_df = run_var_backtest(
                log_returns,
                market_name,
                model_name,
                model_function,
                window=VAR_WINDOW,
            )
            var_rows.append(var_row)

            # Calculate the share of exceeding stocks per day
            daily_shares = var_exceed_df.mean(axis=1)
            var_daily_shares[market_name][model_name] = daily_shares

    simulated_results = simulate_data(alphas=(1.7, 1.9))
    total_results = market_results | simulated_results

    # Plot and save Hill plot
    fig_hill = plot_hill_curves(total_results, single_stocks, k_band=(0.02, 0.05))
    fig_hill.savefig(FIG_PATH / "hill_curve.png", dpi=150)
    plt.close(fig_hill)
    # Plot and save ECDF
    fig_ecdf = plot_ecdf(ecdf_results)
    fig_ecdf.savefig(FIG_PATH / "ecdf.png", dpi=150)
    plt.close(fig_ecdf)
    # Record summary to .md
    summary_hill = pd.DataFrame(summary_rows)
    summary_md = summary_hill.to_markdown(index=False)
    (FIG_PATH / f"summary.md").write_text(summary_md)

    # Record Value-at-Risk summary
    summary_var = pd.DataFrame(var_rows)
    summary_var_md = summary_var.to_markdown(index=False)
    (FIG_PATH / f"var_summary.md").write_text(summary_var_md)

    fig_var = plot_exceedance_over_time(var_daily_shares, smoothing_window=252)
    fig_var.savefig(FIG_PATH / "var_over_time.png", dpi=150)
    plt.close(fig_var)
