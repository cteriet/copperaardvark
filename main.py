"""Entry points for simulating fake data, plotting and fitting a Bayesian model using a config.json and a priors.json."""
import argparse
import json
import pandas as pd
from typing import Dict, Any

from simulation.config import MMMConfig
from simulation.simulation import simulate_mmm_data, calculate_true_sales
from simulation.plotting import plot_mmm_features
from simulation.analysis import (
    build_mmm_model, run_mmm_analysis, 
    plot_parameter_recovery, plot_posterior_predictive,
    plot_total_sales_counterfactual,
    plot_response_curves
)
import arviz as az
import pymc as pm

def load_config(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def cmd_generate(args: argparse.Namespace) -> None:
    config_dict = load_config(args.config)
    config_obj = MMMConfig(**config_dict)
    print("Simulating data...")
    df, true_transforms = simulate_mmm_data(config_obj)
    df.to_csv(args.out_csv, index=False)
    print(f"Data saved to {args.out_csv}")
    
    if args.out_plot:
        plot_mmm_features(
            df=df, 
            channels=config_obj.channels, 
            true_transforms=true_transforms,
            price_per_unit=config_obj.price_per_unit,
            save_path=args.out_plot
        )

def cmd_plot(args: argparse.Namespace) -> None:
    if not args.config:
        raise ValueError("--config is strictly required to supply price_per_unit for plotting.")
        
    config_dict = load_config(args.config)
    df = pd.read_csv(args.csv)
    channels = config_dict['channels']
    price = config_dict['price_per_unit']
    
    # Fail fast if the JSON is missing the key
    if 'price_per_unit' not in config_dict:
        raise KeyError("'price_per_unit' is missing from your config.json file.")
        
    print(f"Generating backup plot from {args.csv}...")
    plot_mmm_features(
        df=df, 
        channels=channels, 
        true_transforms={}, 
        price_per_unit=price,
        save_path=args.out_plot
    )

def cmd_analyze(args: argparse.Namespace) -> None:
    config_dict = load_config(args.config)
    priors_dict = load_config(args.priors) # Load new priors config
    df = pd.read_csv(args.csv)
    
    print("Starting Bayesian inference with PyMC...")
    model, trace = run_mmm_analysis(df, config_dict, priors_dict) # Pass priors down
    
    summary_df = az.summary(trace, round_to=2, ci_prob=0.94, ci_kind='hdi')
    print(summary_df)
    
    plot_parameter_recovery(trace, config_dict, args.out_plot)
    predictive_plot_path = args.out_plot.replace(".png", "_predictive.png")
    plot_posterior_predictive(trace, df, config_dict, predictive_plot_path)
    
    if args.out_trace:
        trace.to_netcdf(args.out_trace)
        print(f"Model trace saved to {args.out_trace}")

def cmd_predict_cf(args: argparse.Namespace) -> None:
    """Command to run a counterfactual prediction on a trained model."""
    config_dict = load_config(args.config)
    priors_dict = load_config(args.priors) 
    df = pd.read_csv(args.csv)
    
    # Instantiate the config object to pass to calculate_true_sales
    config_obj = MMMConfig(**config_dict)
    
    print(f"Loading trace from {args.trace}...")
    trace = az.from_netcdf(args.trace)
    
    # 1. Rebuild the model context 
    model = build_mmm_model(df, config_dict, priors_dict)
    
    # 2. Create the counterfactual dataframe
    df_cf = df.copy()
    channel = args.modify_channel
    
    print(f"Applying {args.multiplier}x multiplier to {channel}...")
    df_cf[channel] = df_cf[channel] * args.multiplier
    
    # Calculate true counterfactual sales using ground truth params
    true_cf_sales = calculate_true_sales(df_cf, config_obj)

    # 3. Run the counterfactual prediction
    print("Generating counterfactual predictions...")
    with model:
        pm.set_data({
            f"{channel}_spend_data": df_cf[channel].values
        })
        
        cf_trace = pm.sample_posterior_predictive(
            trace, 
            predictions=True,
            random_seed=config_dict.get('random_seed', 42)
        )
    
    # 4. Plot the results
    print("Generating counterfactual visuals...")
    
    # Plot weekly results, including hypothetical ground truth scenario and predicted scenario
    plot_posterior_predictive(
        cf_trace, df_cf, config_dict, args.out_plot, 
        group='predictions', true_cf_sales=true_cf_sales
    )
    
    # Generate the total sales distribution plot
    total_plot_path = args.out_plot.replace(".png", "_total.png")
    plot_total_sales_counterfactual(
        cf_trace, df, true_cf_sales, config_dict, total_plot_path, group='predictions'
    )

def cmd_plot_response(args: argparse.Namespace) -> None:
    """Command to generate saturation and profit curves from a trained model."""
    config_dict = load_config(args.config)
    df = pd.read_csv(args.csv)
    
    print(f"Loading trace from {args.trace}...")
    trace = az.from_netcdf(args.trace)
    
    print("Generating Response and Profit curves...")
    plot_response_curves(trace, df, config_dict, args.out_plot)

def main() -> None:
    parser = argparse.ArgumentParser(description="MMM Simulation and Analysis Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generation command
    parser_gen = subparsers.add_parser("generate")
    parser_gen.add_argument("-c", "--config", required=True)
    parser_gen.add_argument("-d", "--out-csv", required=True)
    parser_gen.add_argument("-p", "--out-plot", default=None)


    # Plotting command
    parser_plot = subparsers.add_parser("plot-data")
    parser_plot.add_argument("-d", "--csv", required=True)
    parser_plot.add_argument("-c", "--config", default=None)
    parser_plot.add_argument("-p", "--out-plot", required=True)

    # Analysis command
    parser_ana = subparsers.add_parser("analyze")
    parser_ana.add_argument("-d", "--csv", required=True)
    parser_ana.add_argument("-c", "--config", required=True)
    parser_ana.add_argument("-r", "--priors", required=True, help="Path to JSON file containing model priors")
    parser_ana.add_argument("-p", "--out-plot", required=True)
    parser_ana.add_argument("-t", "--out-trace", default=None, help="Path to save PyMC trace (.nc)")

    # Counterfactual command
    parser_cf = subparsers.add_parser("predict-cf", help="Load trace and predict on counterfactual data.")
    parser_cf.add_argument("-d", "--csv", required=True, help="Base CSV containing latent data")
    parser_cf.add_argument("-c", "--config", required=True, help="Config JSON")
    parser_cf.add_argument("-r", "--priors", required=True, help="Path to JSON file containing model priors")
    parser_cf.add_argument("-t", "--trace", required=True, help="Saved NetCDF trace from analysis")
    parser_cf.add_argument("-p", "--out-plot", required=True, help="Path to save predictive plot")
    parser_cf.add_argument("--modify-channel", type=str, help="Channel to adjust for CF scenario")
    parser_cf.add_argument("--multiplier", type=float, help="Multiplier for the channel (e.g. 1.5 for +50 percent)")

    # Response Curve command
    parser_resp = subparsers.add_parser("plot-response", help="Generate Saturation and Profit curves.")
    parser_resp.add_argument("-d", "--csv", required=True, help="Base CSV containing data")
    parser_resp.add_argument("-c", "--config", required=True, help="Config JSON")
    parser_resp.add_argument("-t", "--trace", required=True, help="Saved NetCDF trace from analysis")
    parser_resp.add_argument("-p", "--out-plot", required=True, help="Path to save the response plots")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "plot-data":
        cmd_plot(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "predict-cf":
        cmd_predict_cf(args)
    elif args.command == "plot-response":
        cmd_plot_response(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()