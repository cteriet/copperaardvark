import pymc as pm
import pytensor
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import arviz as az
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import pymc.sampling.jax as pmjax

import json
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# Meridian Core Imports
from meridian.data import data_frame_input_data_builder
from meridian.model import model, spec
from meridian.analysis import analyzer

def run_meridian_analysis(csv_path: str, config_path: str, out_plot_prefix: str):
    # ---------------------------------------------------------
    # 1. Load Data & Config
    # ---------------------------------------------------------
    df = pd.read_csv(csv_path)
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Meridian often requires 'geo' and 'population' columns
    if "geo" not in df.columns:
        df["geo"] = "national"
    if "population" not in df.columns:
        df["population"] = 1.0

    # --- NEW: Generate a mock date string for Meridian ---
    # We pick an arbitrary start date and add 'week' number of weeks to it.
    start_date = pd.to_datetime("2023-01-01")
    df["mock_date"] = start_date + pd.to_timedelta(df["week"], unit="W")
    df["mock_date"] = df["mock_date"].dt.strftime("%Y-%m-%d")
    # -----------------------------------------------------

    # ---------------------------------------------------------
    # 2. Build Meridian Input Data
    # ---------------------------------------------------------
    media_channels = ["tv_spend", "digital_spend", "radio_spend"]
    control_vars = ["interest_rate", "comp_tv_spend"] 

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue", 
        default_kpi_column="sales"
    )

    builder = (
        builder.with_kpi(df, kpi_col="sales", geo_col="geo", time_col="mock_date")
        .with_media(
            df, 
            media_cols=media_channels, 
            media_spend_cols=media_channels, 
            media_channels=media_channels,  # <-- This was the missing argument
            geo_col="geo", 
            time_col="mock_date"
        )
        .with_controls(df, control_cols=control_vars, geo_col="geo", time_col="mock_date")
        .with_population(df, population_col="population", geo_col="geo")
    )
    
    input_data = builder.build()

    # ---------------------------------------------------------
    # 3. Model Specification & Fitting
    # ---------------------------------------------------------
    # We initialize the default model specification. You can inject custom
    # prior_distribution.PriorDistribution() here if needed.
    model_spec = spec.ModelSpec()
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    
    print("Fitting the Meridian model via NUTS. This may take a few minutes...")
    # Adjust chains and draws based on your hardware (GPU recommended for Meridian)
    mmm.sample_posterior(n_chains=4, n_adapt=1000, n_burnin=1000, n_keep=1000, seed=42)

    # ---------------------------------------------------------
    # 4. Parameter Recovery: Posteriors vs. Ground Truth
    # ---------------------------------------------------------
    trace = mmm.inference_data
    channel_params = config.get("channel_params", {})
    
    fig, axes = plt.subplots(len(media_channels), 2, figsize=(14, 5 * len(media_channels)))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, ch in enumerate(media_channels):
        if ch not in channel_params:
            continue
            
        # Ground truth values from simple_config.json
        gt_theta = channel_params[ch].get("theta")
        gt_alpha = channel_params[ch].get("alpha")
        
        try:
            # Extract samples using ArviZ from Meridian's posterior
            # Note: Parameter names map to Meridian's internal naming schema
            adstock_samples = trace.posterior["adstock_decay"].sel(media_channel=ch).values.flatten()
            hill_samples = trace.posterior["hill_slope"].sel(media_channel=ch).values.flatten()

            # Plot Adstock Decay (Theta)
            az.plot_posterior(adstock_samples, ax=axes[i, 0], point_estimate="mean", ref_val=gt_theta)
            axes[i, 0].set_title(f"{ch} - Adstock Decay (Recovery)")

            # Plot Hill Slope (Alpha)
            az.plot_posterior(hill_samples, ax=axes[i, 1], point_estimate="mean", ref_val=gt_alpha)
            axes[i, 1].set_title(f"{ch} - Hill Slope (Recovery)")
            
        except KeyError as e:
            print(f"Skipping {ch} plots. Trace variable not found: {e}")

    plt.suptitle("Meridian Parameter Recovery vs Ground Truth", fontsize=16)
    plt.savefig(f"{out_plot_prefix}_parameter_recovery.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 5. Response & Profit Curves
    # ---------------------------------------------------------
    print("Generating Response and Profit Curves...")
    
    # Meridian's Analyzer class handles the extraction of expected responses
    mmm_analyzer = analyzer.Analyzer(mmm)
    
    # Extract the response curve data
    # (Meridian provides standard evaluation grids for this)
    response_data = mmm_analyzer.expected_response_curves()
    
    fig_rc, ax_rc = plt.subplots(figsize=(10, 6))
    for ch in media_channels:
        # Plotting the mean expected response across the spend grid
        spend_grid = response_data[ch]["spend_grid"]
        response_mean = response_data[ch]["response_mean"]
        
        ax_rc.plot(spend_grid, response_mean, label=f"{ch} Response", linewidth=2)
        
    ax_rc.set_title("Media Response Curves (Saturation)")
    ax_rc.set_xlabel("Spend ($)")
    ax_rc.set_ylabel("Incremental Sales")
    ax_rc.legend()
    ax_rc.grid(True, alpha=0.3)
    
    plt.savefig(f"{out_plot_prefix}_response_curves.png", bbox_inches="tight")
    plt.close()

    print(f"Done! Plots saved with prefix: {out_plot_prefix}")
    return mmm

def geometric_adstock_pt(spend: pt.TensorVariable, theta: pt.TensorVariable) -> pt.TensorVariable:
    """PyTensor implementation of geometric adstock using scan."""
    def step(x_t: pt.TensorVariable, y_tm1: pt.TensorVariable, theta: pt.TensorVariable) -> pt.TensorVariable:
        return x_t + theta * y_tm1
    
    outputs, _ = pytensor.scan(
        fn=step,
        sequences=[spend],
        outputs_info=[pt.as_tensor_variable(np.array(0.0, dtype=spend.dtype))],
        non_sequences=[theta],
        strict=True
    )
    return outputs

def hill_function_pt(x: pt.TensorVariable, alpha: pt.TensorVariable, gamma: pt.TensorVariable) -> pt.TensorVariable:
    """PyTensor implementation of the Hill function."""
    epsilon = 1e-8
    x_safe = pt.clip(x, epsilon, np.inf)
    return 1.0 / (1.0 + (gamma / x_safe) ** alpha)

def create_prior(name: str, prior_spec: Dict[str, Any]):
    """Helper to dynamically instantiate PyMC distributions from config."""
    dist_class = getattr(pm, prior_spec['dist'])
    return dist_class(name, **prior_spec['kwargs'])

def build_mmm_model(df: pd.DataFrame, config_dict: Dict[str, Any], priors_dict: Dict[str, Any]) -> pm.Model:
    n_weeks: int = len(df)
    time_idx: np.ndarray = np.arange(n_weeks)
    
    sales: np.ndarray = df[config_dict['col_target']].values
    interest_rate: np.ndarray = df[config_dict['col_interest_rate']].values
    is_cheapest: np.ndarray = df[config_dict['col_cheapest_flag']].values
    channels: list[str] = config_dict['channels']

    p_global = priors_dict["global"]

    with pm.Model() as mmm:
        time_data = pm.Data("time_data", time_idx)
        interest_data = pm.Data("interest_data", interest_rate)
        cheapest_data = pm.Data("cheapest_data", is_cheapest)
        
        spend_vars = {
            ch: pm.Data(f"{ch}_spend_data", df[ch].values) 
            for ch in channels
        }

        # Latent Trend & Seasonality
        sigma_trend = create_prior("sigma_trend", p_global["sigma_trend"])
        
        # Init dist for GaussianRandomWalk is special as it needs `.dist()`
        init_spec = p_global["trend_init"]
        init_dist = getattr(pm, init_spec['dist']).dist(**init_spec['kwargs'])
        
        trend = pm.GaussianRandomWalk("trend", sigma=sigma_trend, shape=n_weeks, init_dist=init_dist)
        
        amp_4w = create_prior("amp_4w", p_global["amp_4w"])
        amp_52w = create_prior("amp_52w", p_global["amp_52w"])
        seasonality = amp_4w * pt.sin(2 * np.pi * time_data / 4) + amp_52w * pt.sin(2 * np.pi * time_data / 52)
        
        # Controls
        coef_interest = create_prior("coef_interest", p_global["coef_interest"])
        control_effect = coef_interest * interest_data
        
        media_effects = []
        for ch in channels:
            p_ch = priors_dict["channels"][ch]
            
            theta = create_prior(f"{ch}_theta", p_ch["theta"])
            alpha = create_prior(f"{ch}_alpha", p_ch["alpha"])
            gamma = create_prior(f"{ch}_gamma", p_ch["gamma"])
            
            coef_cheap = create_prior(f"{ch}_coef_cheap", p_ch["coef_cheap"])
            coef_not_cheap = create_prior(f"{ch}_coef_not_cheap", p_ch["coef_not_cheap"])
            
            adstocked = geometric_adstock_pt(spend_vars[ch], theta)
            saturated = hill_function_pt(adstocked, alpha, gamma)
            
            active_coef = pt.switch(cheapest_data, coef_cheap, coef_not_cheap)
            channel_effect = pm.Deterministic(f"{ch}_effect", saturated * active_coef)
            media_effects.append(channel_effect)
            
        total_media = pm.math.sum(media_effects, axis=0)
        
        # Likelihood
        mu = trend + seasonality + control_effect + total_media
        mu_safe = pt.clip(mu, 0.1, np.inf) 
        sigma_obs = create_prior("sigma_obs", p_global["sigma_obs"])
        
        pm.Normal("obs", mu=mu_safe, sigma=sigma_obs, observed=sales)
        
    return mmm

def run_mmm_analysis(df: pd.DataFrame, config_dict: Dict[str, Any], priors_dict: Dict[str, Any]) -> Tuple[pm.Model, az.InferenceData]:
    """Runs the MCMC sampling using the Numpyro backend."""
    mmm = build_mmm_model(df, config_dict, priors_dict)
    
    with mmm:
        trace = pmjax.sample_numpyro_nuts(
            draws=1500, 
            tune=1000, 
            chains=4, 
            target_accept=0.95, 
            random_seed=config_dict.get('random_seed', 42)
        )
        
        pm.sample_posterior_predictive(
            trace, 
            extend_inferencedata=True, 
            random_seed=config_dict.get('random_seed', 42)
        )
        
    return mmm, trace

def plot_parameter_recovery(trace: az.InferenceData, config_dict: Dict[str, Any], save_path: str) -> None:
    """Plots posterior estimates against ground-truth configuration values and saves to disk."""
    channels: list[str] = config_dict['channels']
    config_params: Dict[str, Any] = config_dict['channel_params']
    
    # Setup plotting grids
    metrics: list[str] = ["theta", "alpha", "gamma", "coef_if_cheapest", "coef_if_not_cheapest"]
    name_mapping: Dict[str, str] = {"coef_if_cheapest": "coef_cheap", "coef_if_not_cheapest": "coef_not_cheap"}

    # We add +1 to rows for the "Global Parameters" row
    fig, axes = plt.subplots(len(channels) + 1, len(metrics), figsize=(20, 3 * (len(channels) + 1)))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    # --- Plot Global Parameters (Row 0) ---
    global_vars = ["coef_interest", "amp_4w", "amp_52w"]
    true_globals = {
        "coef_interest": config_dict["interest_rate_coef"],
        "amp_4w": config_dict["seasonality_amplitude_4w"],
        "amp_52w": config_dict["seasonality_amplitude_52w"]
    }
    
    for j, g_var in enumerate(global_vars):
        ax = axes[0, j]
        posterior_samples = trace.posterior[g_var].values.flatten()
        ax.hist(posterior_samples, bins=40, density=True, color='lightgreen', alpha=0.7, label='Posterior')
        ax.axvline(true_globals[g_var], color='red', linestyle='--', linewidth=2, label='True Value')
        ax.set_title(f"GLOBAL\n{g_var}")
        if j == 0: ax.legend()
        ax.set_yticks([]) 
        
    # Hide unused subplots in the first row
    for j in range(len(global_vars), len(metrics)):
        axes[0, j].axis('off')

    # --- Plot Channel Parameters (Rows 1+) ---
    for i, ch in enumerate(channels):
        row_idx = i + 1
        for j, metric in enumerate(metrics):
            ax = axes[row_idx, j]
            trace_suffix = name_mapping.get(metric, metric)
            var_name = f"{ch}_{trace_suffix}"
            
            posterior_samples = trace.posterior[var_name].values.flatten()
            ax.hist(posterior_samples, bins=40, density=True, color='skyblue', alpha=0.7, label='Posterior')
            true_val = config_params[ch][metric]
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='True Value')
            ax.set_title(f"{ch}\n{metric}")
            ax.set_yticks([]) 
            
    plt.suptitle("Parameter Recovery: Ground Truth vs. Estimated Posterior", fontsize=18, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Recovery plot saved to {save_path}")
    plt.close()

def plot_posterior_predictive(
    trace: az.InferenceData, 
    df: pd.DataFrame, 
    config_dict: Dict[str, Any], 
    save_path: str, 
    group: str = 'posterior_predictive',
    true_cf_sales: np.ndarray = None  # <--- NEW ARGUMENT
) -> None:
    sales = df[config_dict['col_target']].values
    time_idx = df[config_dict['col_time']].values
    
    # Dynamically select the group based on the parameter
    pp_sales = getattr(trace, group)['obs']
    
    sales_mean = pp_sales.mean(dim=['chain', 'draw'])
    sales_hdi = az.hdi(pp_sales, prob=0.95)
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_idx, sales, color='black', label='Observed Sales', linewidth=2)
    plt.plot(time_idx, sales_mean, color='orange', label='Estimated Sales (Mean)', linestyle='--', linewidth=2)
    
    # <--- NEW: Plot True CF Sales if provided --->
    if true_cf_sales is not None:
        plt.plot(time_idx, true_cf_sales, color='green', label='True Counterfactual Sales', linestyle='-.', linewidth=2)
    
    # Safely handle the HDI dimension name depending on the Arviz version
    dim_name = 'ci_bound' if 'ci_bound' in sales_hdi.dims else 'hdi'
    
    plt.fill_between(time_idx, 
                     sales_hdi.isel({dim_name: 0}), 
                     sales_hdi.isel({dim_name: 1}), 
                     color='orange', alpha=0.3, label='95% Credible Interval')
    
    price = config_dict['price_per_unit']
    ax = plt.gca()
    sec_ax = ax.secondary_yaxis('right', functions=(lambda x: x * price, lambda x: x / price))
    sec_ax.set_ylabel('Money Earned ($)', fontsize=12)
    
    plt.title('Model Fit: Observed Sales vs Posterior Predictive', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Posterior predictive plot saved to {save_path}")
    plt.close()

def plot_total_sales_counterfactual(
    trace: az.InferenceData,
    df_original: pd.DataFrame,
    true_cf_sales: np.ndarray,
    config_dict: Dict[str, Any],
    save_path: str,
    group: str = 'predictions'
) -> None:
    """Plots the posterior distribution of total predicted sales against observed and true counterfactual totals."""
    # 1. Calculate deterministic totals
    obs_total = df_original[config_dict['col_target']].sum()
    true_cf_total = true_cf_sales.sum()

    # 2. Extract distribution of predicted totals
    pp_sales = getattr(trace, group)['obs']
    # Dynamically find the time dimension to sum over (ignoring chain and draw)
    time_dim = [d for d in pp_sales.dims if d not in ('chain', 'draw')][0]
    pred_totals = pp_sales.sum(dim=time_dim).values.flatten()

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(pred_totals, bins=40, density=True, alpha=0.7, color='skyblue', label='Predicted CF Total Sales (Distribution)')

    plt.axvline(obs_total, color='black', linestyle='-', linewidth=2, label=f'Observed Total ({obs_total:.0f})')
    plt.axvline(true_cf_total, color='green', linestyle='--', linewidth=2, label=f'True CF Total ({true_cf_total:.0f})')

    # Add 95% HDI lines for the predicted total
    lower, upper = np.percentile(pred_totals, [2.5, 97.5])
    plt.axvline(lower, color='orange', linestyle=':', linewidth=2, label='95% HDI (Predicted)')
    plt.axvline(upper, color='orange', linestyle=':', linewidth=2)

    price = config_dict['price_per_unit']
    ax = plt.gca()
    sec_ax = ax.secondary_yaxis('right', functions=(lambda x: x * price, lambda x: x / price))
    sec_ax.set_ylabel('Money Earned ($)', fontsize=12)

    plt.title('Total Sales: Observed vs. True Counterfactual vs. Predicted Distribution', fontsize=14)
    plt.xlabel('Total Sales', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Total sales counterfactual plot saved to {save_path}")
    plt.close()


def plot_response_curves(trace: az.InferenceData, df: pd.DataFrame, config_dict: Dict[str, Any], save_path: str) -> None:
    """Plots Saturation (Response) and Profit (ROI) curves for each channel, with a marginal spend density plot."""
    channels = config_dict['channels']
    price = config_dict['price_per_unit']
    config_params = config_dict['channel_params']
    
    # Calculate the historical probability of being the cheapest to weight the coefficients
    p_cheap = df[config_dict['col_cheapest_flag']].mean()
    
    # Use GridSpec for complex layouts (Main plot + marginal density plot)
    fig = plt.figure(figsize=(16, 6 * len(channels)))
    outer_gs = gridspec.GridSpec(len(channels), 2, hspace=0.4, wspace=0.3)
    
    for i, ch in enumerate(channels):
        # --- Layout Setup ---
        # 80% height for the main curve, 20% for the KDE plot, minimal vertical space
        inner_gs_resp = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i, 0], height_ratios=[4, 1], hspace=0.05)
        ax_resp = fig.add_subplot(inner_gs_resp[0, 0])
        ax_resp_kde = fig.add_subplot(inner_gs_resp[1, 0], sharex=ax_resp)
        
        inner_gs_prof = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i, 1], height_ratios=[4, 1], hspace=0.05)
        ax_prof = fig.add_subplot(inner_gs_prof[0, 0])
        ax_prof_kde = fig.add_subplot(inner_gs_prof[1, 0], sharex=ax_prof)
        
        # --- Data Prep ---
        max_spend = df[ch].max()
        x_range = np.linspace(0, max_spend * 1.5, 100)
        hist_spend = df[ch].values

        # Remove the zero spend days from the KDE plot
        hist_spend = hist_spend[hist_spend > 0]
        
        # Calculate KDE for the historical spend distribution
        try:
            kde = gaussian_kde(hist_spend)
            kde_y = kde(x_range)
        except np.linalg.LinAlgError:
            # Fallback if there is zero variance in the spend (e.g., all 0s)
            kde_y = np.zeros_like(x_range)
        
        # --- Extract Posterior & Config Parameters ---
        alpha = trace.posterior[f"{ch}_alpha"].values.flatten()[:, None]
        gamma = trace.posterior[f"{ch}_gamma"].values.flatten()[:, None]
        coef_c = trace.posterior[f"{ch}_coef_cheap"].values.flatten()[:, None]
        coef_nc = trace.posterior[f"{ch}_coef_not_cheap"].values.flatten()[:, None]
        
        # Blend the coefficient based on state probability
        coef = (p_cheap * coef_c) + ((1 - p_cheap) * coef_nc)
        
        true_alpha = config_params[ch]['alpha']
        true_gamma = config_params[ch]['gamma']
        true_coef_c = config_params[ch]['coef_if_cheapest']
        true_coef_nc = config_params[ch]['coef_if_not_cheapest']
        true_coef = (p_cheap * true_coef_c) + ((1 - p_cheap) * true_coef_nc)
        
        # --- Calculate Curves ---
        epsilon = 1e-8
        x_safe = np.clip(x_range, epsilon, None)
        
        response_draws = coef * (1.0 / (1.0 + (gamma / x_safe)**alpha))
        profit_draws = (response_draws * price) - x_range
        
        resp_mean = response_draws.mean(axis=0)
        resp_lower = np.percentile(response_draws, 2.5, axis=0)
        resp_upper = np.percentile(response_draws, 97.5, axis=0)
        
        prof_mean = profit_draws.mean(axis=0)
        prof_lower = np.percentile(profit_draws, 2.5, axis=0)
        prof_upper = np.percentile(profit_draws, 97.5, axis=0)
        
        true_response = true_coef * (1.0 / (1.0 + (true_gamma / x_safe)**true_alpha))
        true_profit = (true_response * price) - x_range
        
        # ==========================================
        # 1. Plot Response (Saturation)
        # ==========================================
        ax_resp.plot(x_range, resp_mean, color='blue', linewidth=2, label='Mean Response')
        ax_resp.plot(x_range, true_response, color='purple', linestyle='--', linewidth=2, label='True Response')
        ax_resp.fill_between(x_range, resp_lower, resp_upper, color='blue', alpha=0.2, label='95% HDI')
        
        ax_resp.set_title(f"{ch}: Saturation Curve")
        ax_resp.set_ylabel("Incremental Sales (Units)")
        ax_resp.legend(loc='upper left')
        ax_resp.grid(alpha=0.3)
        plt.setp(ax_resp.get_xticklabels(), visible=False) # Hide x-ticks for the main plot
        
        # Secondary axis for Money
        sec_ax_resp = ax_resp.secondary_yaxis('right', functions=(lambda x: x * price, lambda x: x / price))
        sec_ax_resp.set_ylabel("Incremental Revenue ($)")
        
        # Plot KDE for Response
        ax_resp_kde.fill_between(x_range, 0, kde_y, color='gray', alpha=0.4)
        ax_resp_kde.plot(x_range, kde_y, color='black', alpha=0.6, linewidth=1)
        ax_resp_kde.set_xlabel("Money Invested ($)")
        ax_resp_kde.set_ylabel("Density")
        ax_resp_kde.set_yticks([]) # Hide the y-axis ticks as it's just a visual reference
        
        # ==========================================
        # 2. Plot Profit (ROI)
        # ==========================================
        ax_prof.plot(x_range, prof_mean, color='green', linewidth=2, label='Mean Profit')
        ax_prof.plot(x_range, true_profit, color='purple', linestyle='--', linewidth=2, label='True Profit')
        ax_prof.fill_between(x_range, prof_lower, prof_upper, color='green', alpha=0.2, label='95% HDI')
        ax_prof.axhline(0, color='red', linestyle='--', linewidth=1) # Break-even line
        
        ax_prof.set_title(f"{ch}: Profit (ROI) Curve")
        ax_prof.set_ylabel("Profit ($)")
        ax_prof.legend(loc='upper left')
        ax_prof.grid(alpha=0.3)
        plt.setp(ax_prof.get_xticklabels(), visible=False) # Hide x-ticks for the main plot
        
        # Plot KDE for Profit
        ax_prof_kde.fill_between(x_range, 0, kde_y, color='gray', alpha=0.4)
        ax_prof_kde.plot(x_range, kde_y, color='black', alpha=0.6, linewidth=1)
        ax_prof_kde.set_xlabel("Money Invested ($)")
        ax_prof_kde.set_ylabel("Density")
        ax_prof_kde.set_yticks([]) # Hide the y-axis ticks
        
    # We no longer use plt.tight_layout() here because it conflicts with GridSpec spacing
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Response and Profit curves saved to {save_path}")
    plt.close()