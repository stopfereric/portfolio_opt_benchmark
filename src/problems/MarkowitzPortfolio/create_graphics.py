import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from src.problems.MarkowitzPortfolio.create_random_markowitz_config_files import \
    get_random_portfolio, calc_return, calc_volatility
from src.run_multiple_optimizations import Benchmark


def analyze_random_portfolio_statistics_minvola(base_markowitz_configs_path: str,
                                                stock_market: str = "nasdaq",
                                                num_random_portfolios: int = 100):
    '''
    Analyzes the mean return and mean variance of random portfolios for different instance sizes
    of minvola instances from a specific stock market. Creates boxplots showing the distribution
    of returns and volatilities across different instance sizes.

    Parameters:
    -----------
    base_markowitz_configs_path : str
        Path to the base MarkowitzPortfolio config directory containing the from_nasdaq folder
    stock_market : str
        The stock market name (e.g., 'nasdaq')
    num_random_portfolios : int
        Number of random portfolios to generate for each instance (default: 100)
    '''

    # Load stock market data once
    stockinfo_directory = os.path.dirname(os.path.abspath(__file__))
    asset_returns_df = pd.read_csv(
        os.path.join(stockinfo_directory, f'{stock_market}_annual_returns.csv'), delimiter='\t')
    covariance_matrix_df = pd.read_csv(
        os.path.join(stockinfo_directory, f'{stock_market}_annualized_covariance_matrix.csv'),
        delimiter='\t')

    # Dictionary to store returns and volatilities for each instance size
    returns_by_size = defaultdict(list)
    volatilities_by_size = defaultdict(list)

    # Navigate to from_nasdaq folder
    from_market_path = os.path.join(base_markowitz_configs_path, f"from_{stock_market}")

    if not os.path.exists(from_market_path):
        raise FileNotFoundError(f"Path {from_market_path} does not exist")

    print(f"Analyzing minvola instances in {from_market_path}...")

    # Iterate through subdirectories representing different instance sizes
    for size_dir in os.listdir(from_market_path):
        size_path = os.path.join(from_market_path, size_dir)

        if not os.path.isdir(size_path):
            continue

        # Extract number of assets from directory name (e.g., "3assets" -> 3)
        try:
            num_assets = int(size_dir.replace("assets", "").strip())
        except (ValueError, AttributeError):
            print(f"Warning: Could not extract asset number from directory name: {size_dir}")
            continue

        print(f"  Processing {size_dir}...")

        # Find all minvola config files in this directory
        minvola_files = [f for f in os.listdir(size_path) if f.endswith('.json') and 'minvola' in f and "2000" not in f]

        if not minvola_files:
            print(f"    No minvola files found in {size_dir}")
            continue

        # Process each minvola config file
        for config_file in minvola_files:
            config_path = os.path.join(size_path, config_file)

            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                # Extract the list of chosen assets
                chosen_assets = config_data['problem_config'].get(f'{stock_market}_assets', [])

                if not chosen_assets:
                    print(f"    Warning: No {stock_market}_assets found in {config_file}")
                    continue

                # Get asset returns and covariances for chosen assets
                asset_returns = asset_returns_df[chosen_assets]
                asset_covariances = covariance_matrix_df.loc[chosen_assets, chosen_assets]

                # Generate random portfolios and compute returns and volatilities
                for _ in range(num_random_portfolios):
                    portfolio_weights = get_random_portfolio(len(chosen_assets))
                    returns = calc_return(portfolio_weights, asset_returns)
                    volatility = calc_volatility(portfolio_weights, asset_covariances)

                    returns_by_size[num_assets].append(returns)
                    volatilities_by_size[num_assets].append(volatility)

            except Exception as e:
                print(f"    Error processing {config_file}: {str(e)}")
                continue

    # Sort by instance size for plotting
    sorted_sizes = sorted(returns_by_size.keys())

    if not sorted_sizes:
        print("No data collected. Please check the paths and file contents.")
        return

    # Prepare data for boxplots
    returns_data = [returns_by_size[size] for size in sorted_sizes]
    volatilities_data = [volatilities_by_size[size] for size in sorted_sizes]

    # Plot 1: Returns boxplot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bp1 = ax1.boxplot(returns_data, labels=[str(s) for s in sorted_sizes], patch_artist=True)
    ax1.set_xlabel('Number of Assets', fontsize=14)
    ax1.set_ylabel('Return', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Color the boxes
    for patch in bp1['boxes']:
        patch.set_facecolor('blue')

    plt.tight_layout()

    # Save the returns figure
    output_path_returns = os.path.join(base_markowitz_configs_path,
                                       f"random_portfolio_returns_{stock_market}_minvola.png")
    plt.savefig(output_path_returns, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path_returns}")
    plt.close(fig1)

    # Plot 2: Volatilities boxplot with logarithmic scale
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bp2 = ax2.boxplot(volatilities_data, labels=[str(s) for s in sorted_sizes], patch_artist=True)
    ax2.set_xlabel('Number of Assets', fontsize=14)
    ax2.set_ylabel('Variance (Log Scale)', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')

    # Color the boxes
    for patch in bp2['boxes']:
        patch.set_facecolor('blue')

    plt.tight_layout()

    # Save the volatilities figure
    output_path_volatilities = os.path.join(base_markowitz_configs_path,
                                            f"random_portfolio_volatilities_{stock_market}_minvola.png")
    plt.savefig(output_path_volatilities, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path_volatilities}")
    plt.close(fig2)

    # Also display some statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for size in sorted_sizes:
        avg_return = np.mean(returns_by_size[size])
        avg_volatility = np.mean(volatilities_by_size[size])
        std_return = np.std(returns_by_size[size])
        std_volatility = np.std(volatilities_by_size[size])

        print(f"\nInstance Size: {size} assets")
        print(f"  Returns     - Mean: {avg_return:.6f}, Std: {std_return:.6f}")
        print(f"  Volatilities - Mean: {avg_volatility:.6f}, Std: {std_volatility:.6f}")


def plot_optimal_volatility_by_instance_size(results_dict, output_path=None, boxplot=True):
    grouped_values = {}

    for path, value in results_dict.items():
        match = re.search(r'\\(\d+)assets\\', path)
        if match:
            num_assets = int(match.group(1))

            if num_assets not in grouped_values:
                grouped_values[num_assets] = []

            grouped_values[num_assets].append(value)

    asset_sizes = sorted(grouped_values.keys())


    plt.figure(figsize=(10, 6))
    if boxplot:
        boxplot_data = [grouped_values[size] for size in asset_sizes]
        bp = plt.boxplot(
            boxplot_data,
            labels=[str(size) for size in asset_sizes],
            patch_artist=True
        )
        for box in bp["boxes"]:
            box.set_facecolor("blue")
    else:
        means = [np.mean(grouped_values[size]) for size in asset_sizes]
        stds = [np.std(grouped_values[size], ddof=1) for size in asset_sizes]
        plt.errorbar(
            asset_sizes,
            means,
            yerr=stds,
            fmt='o-',
            capsize=5
        )

    plt.xlabel("Number of Assets")
    plt.ylabel("Optimal Variance")
    plt.title("Optimal Variance vs. Instance Size")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    if output_path:
        plt.savefig(os.path.join(output_path,"distribution_optimal_obj_values.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_random_portfolios_volatility_return_and_efficient_frontier(
    instance_path: str,
    num_random_portfolios: int = 1000,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    with_efficient_frontier: bool = False,
    output_path: str | None = os.path.dirname(__file__),
):
    with open(instance_path, "r") as f:
        config_data = json.load(f)

    chosen_assets = config_data["problem_config"]["nasdaq_assets"]

    stockinfo_directory = os.path.dirname(__file__)

    asset_returns_df = pd.read_csv(
        os.path.join(stockinfo_directory, "nasdaq_annual_returns.csv"),
        delimiter="\t",
    )

    covariance_matrix_df = pd.read_csv(
        os.path.join(stockinfo_directory, "nasdaq_annualized_covariance_matrix.csv"),
        delimiter="\t",
    )

    asset_returns = asset_returns_df[chosen_assets]
    asset_covariances = covariance_matrix_df.loc[chosen_assets, chosen_assets]

    points = []

    for _ in range(num_random_portfolios):
        weights = get_random_portfolio(len(chosen_assets))
        v = calc_volatility(weights, asset_covariances)
        r = calc_return(weights, asset_returns)
        points.append((v, r))

    points = np.array(points)
    vol = points[:, 0]
    ret = points[:, 1]

    gmv_idx = np.argmin(vol)
    gmv_vol = vol[gmv_idx]
    gmv_ret = ret[gmv_idx]

    lower = [(v, r) for v, r in points if r <= gmv_ret]
    upper = [(v, r) for v, r in points if r > gmv_ret]

    lower.sort(key=lambda x: x[1])
    upper.sort(key=lambda x: x[1], reverse=True)

    def frontier(sorted_points):
        res = []
        best_vol = float("inf")
        for v, r in sorted_points:
            if v < best_vol:
                res.append((v, r))
                best_vol = v
        return np.array(res)

    if with_efficient_frontier:
        lower_f = frontier(lower)
        upper_f = frontier(upper)
    else:
        lower_f = []
        upper_f = []

    plt.figure(figsize=(10, 6), dpi=300)

    plt.scatter(vol, ret, s=10, alpha=0.25, label="Random Portfolios")

    plt.scatter(
        gmv_vol,
        gmv_ret,
        color="green",
        s=100,
        zorder=5,
        label="Global Minimum Variance Portfolio",
    )

    if len(upper_f) > 0:
        plt.plot(
            upper_f[:, 0],
            upper_f[:, 1],
            color="green",
            linewidth=2,
            label="Efficient Frontier",
        )

    if len(lower_f) > 0:
        plt.plot(
            lower_f[:, 0],
            lower_f[:, 1],
            color="red",
            linewidth=2,
            label="Other variance-minimal portfolios",
        )


    plt.xlabel("Volatility", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.title(f"Random Portfolios {'with Efficient Frontier' if with_efficient_frontier else ''}", fontsize=14)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tick_params(axis="both", labelsize=10)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            os.path.join(output_path, "ret_vs_vola.png"),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()

def plot_efficient_frontier(output_path: str | None = os.path.dirname(__file__)):
    # Return range
    ret_min = -0.1
    ret_max = 0.18
    ret_gmv = 0.04

    returns = np.linspace(ret_min, ret_max, 500)

    # Right-opening parabola:
    # volatility = a * (return - ret_gmv)^2 + sigma_gmv
    sigma_gmv = 0.08
    a = 8.0

    volatility = sigma_gmv + a * (returns - ret_gmv) ** 2

    inefficient = returns <= ret_gmv
    efficient = returns >= ret_gmv

    plt.figure(figsize=(8, 5))

    plt.plot(
        volatility[efficient],
        returns[efficient],
        color="black",
        linewidth=2,
        label="Efficient frontier",
    )

    plt.plot(
        volatility[inefficient],
        returns[inefficient],
        color="red",
        linewidth=2,
        label="Other variance-minimal portfolios",
    )

    plt.scatter(
        sigma_gmv,
        ret_gmv,
        color="black",
        s=60,
        zorder=5,
        label="Global minimum variance portfolio",
    )

    plt.xlabel("Variance")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            os.path.join(output_path, "efficient_frontier.png"),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()


analyze_random_portfolio_statistics = True
analyze_optimal_volatilities = False
analyze_ret_vs_vola_for_instance = False
just_efficient_frontier = False

if __name__ == "__main__":

    if analyze_random_portfolio_statistics:
        analyze_random_portfolio_statistics_minvola(
            base_markowitz_configs_path=r"C:\Users\stopfer\Documents\git\quopt\config\config_files\problem_configs\MarkowitzPortfolio",
            stock_market="nasdaq",
            num_random_portfolios=100
        )

    if analyze_optimal_volatilities:
        config_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config", "config_files", "benchmark_config.json"
        )

        existing_results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "results",
            "benchmarkrun_portfolioopt_heuristics_new"
        )
        assert os.path.exists(
            existing_results_path), f"results path '{existing_results_path}' does not exist. You must specify in the code."

        benchmark = Benchmark(config_file_path)
        benchmark.create_benchmark_export_folder(existing_results_path)
        opt_sol_info = benchmark.get_all_optimal_solutions_in_export_path()
        plot_optimal_volatility_by_instance_size(opt_sol_info, os.path.dirname(__file__), boxplot=True)

    if analyze_ret_vs_vola_for_instance:
        plot_random_portfolios_volatility_return_and_efficient_frontier(
            instance_path=r"C:\Users\stopfer\Documents\git\quopt\config\config_files\problem_configs\MarkowitzPortfolio\from_nasdaq\40assets\random_MarkowitzPortfolio_fromnasdaq_minvola_40assets_4.json",
            num_random_portfolios=100000,
            xlim=(0, 0.5),
            ylim=(-0.3, 0.1)
        )

    if just_efficient_frontier:
        plot_efficient_frontier()