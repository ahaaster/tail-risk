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
from tail_risk.core import christoffersen_test, rolling_sigma, ewma_sigma
from tail_risk.core import NU_GRID, fit_nu_per_block
from tail_risk.simulation import simulate_data
from tail_risk.plots import plot_hill_curves, plot_ecdf, plot_exceedance_over_time
from tail_risk.plots import plot_exceedance_summary

jax.config.update("jax_enable_x64", True)

WORKING_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WORKING_DIR / "data"
FIG_PATH = WORKING_DIR / "figures"
README_PATH = WORKING_DIR / "README.md"
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

    value_at_risk = model_function(log_returns, window=window, level=level)
    exceedances = losses > value_at_risk

    forecast_count = exceedances.shape[0]
    exceed_count = exceedances.sum(axis=0)

    _, p_val_kupiec = kupiec_test(exceed_count, forecast_count, level)
    _, p_val_chris = christoffersen_test(exceedances)
    rejections_kupiec = (p_val_kupiec < 0.05).mean()
    rejections_chris = (p_val_chris < 0.05).mean()

    exceed_rate = np.asarray(exceed_count / forecast_count)
    quarter, median, three_quart = calc_quartiles(exceed_rate)

    results = {
        "market": MARKET_LABELS[market_name],
        "model": MODEL_LABELS[model_name],
        "median rate exceeded": f"{median:.2%}",
        "IQR-band": f"{quarter:.2%}-{three_quart:.2%}",
        "share rejected coverage (5%)": f"{rejections_kupiec:.1%}",
        "share rejected indep. (5%)": f"{rejections_chris:.1%}",
    }

    dates = log_returns.index[window:]
    exceedance_df = pd.DataFrame(np.asarray(exceedances), index=dates)
    return results, exceed_rate, exceedance_df


def summarise_nu(log_returns: pd.DataFrame, market_name: str) -> list:
    sigma_methods = {"rolling": rolling_sigma, "EWMA": ewma_sigma}
    rows = []
    for sigma_name, sigma_function in sigma_methods.items():
        _, nu_index_per_block = fit_nu_per_block(
            log_returns, sigma_function, VAR_WINDOW
        )
        nu_values = np.asarray(NU_GRID[nu_index_per_block])
        at_grid_edge = (nu_values == NU_GRID[0]) | (nu_values == NU_GRID[-1])

        median_overall = np.median(nu_values)
        first_median, last_median = np.median(nu_values[0]), np.median(nu_values[-1])
        rows.append(
            {
                "market": MARKET_LABELS[market_name],
                r"$\sigma$ method": sigma_name,
                r"median $\nu$ (all blocks)": f"{median_overall:.1f}",
                r"median $\nu$ first -> last block": f"{first_median:.1f} -> {last_median:.1f}",
                "share at grid edge": f"{at_grid_edge.mean():.1%}",
            }
        )
    return rows


def write_table_into_readme(table_name: str, table_markdown: str) -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")

    start_marker, end_marker = (
        f"<!-- {table_name}:start -->",
        f"<!-- {table_name}:end -->",
    )
    before, rest = readme_text.split(start_marker)
    _, after = rest.split(end_marker)

    README_PATH.write_text(
        f"{before}{start_marker}\n{table_markdown}\n{end_marker}{after}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    DATA_PATH.mkdir(exist_ok=True)
    FIG_PATH.mkdir(exist_ok=True)
    data_paths = DATA_PATH.glob("logret_*.csv")

    market_results, ecdf_results, var_daily_shares, var_rates = {}, {}, {}, {}
    single_stocks, summary_rows, var_rows, nu_rows = [], [], [], []
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
        var_daily_shares[market_name], var_rates[market_name] = {}, {}
        for model_name, model_function in VAR_MODELS.items():
            var_row, exceed_rate, var_exceed_df = run_var_backtest(
                log_returns,
                market_name,
                model_name,
                model_function,
                window=VAR_WINDOW,
            )
            var_rows.append(var_row)
            var_rates[market_name][model_name] = exceed_rate
            # Calculate the share of exceeding stocks per day
            daily_shares = var_exceed_df.mean(axis=1)
            var_daily_shares[market_name][model_name] = daily_shares

        nu_summary = summarise_nu(log_returns, market_name)
        nu_rows.extend(nu_summary)

        # comparison = var_exceed_df["t_rolling"] != var_exceed_df["t_ewma"]
        # print(market_name, comparison.to_numpy().mean())

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
    hill_summary_md = summary_hill.to_markdown(index=False)
    (FIG_PATH / f"hill_summary.md").write_text(hill_summary_md)
    write_table_into_readme("hill_summary_table", hill_summary_md)

    # Record Value-at-Risk summary
    summary_var = pd.DataFrame(var_rows)
    summary_var = summary_var.to_markdown(index=False)
    (FIG_PATH / f"var_summary.md").write_text(summary_var)
    write_table_into_readme("var_summary_table", summary_var)
    # plot Value-at-Risk
    fig_var = plot_exceedance_over_time(var_daily_shares, smoothing_window=252)
    fig_var.savefig(FIG_PATH / "var_over_time.png", dpi=150)
    plt.close(fig_var)
    # Record summary of nu-fit
    summary_nu = pd.DataFrame(nu_rows)
    summary_nu = summary_nu.to_markdown(index=False)
    (FIG_PATH / f"nu_summary.md").write_text(summary_nu)
    # Summarise plot
    fig_summary = plot_exceedance_summary(var_rates)
    fig_summary.savefig(FIG_PATH / "var_summary.png", dpi=150)
    plt.close(fig_summary)
