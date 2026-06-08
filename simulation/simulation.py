import numpy as np
import pandas as pd
from typing import Dict, Tuple
from simulation.config import MMMConfig
from simulation.generators import (
    generate_spiky_spend, generate_trend, generate_seasonality, 
    generate_interest_rate, generate_cheapest_flag
)
from simulation.transformation import geometric_adstock, hill_function

def simulate_mmm_data(config: MMMConfig) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.random_seed)
    n = config.n_weeks
    
    # 1. Initialize DataFrame
    df = pd.DataFrame({config.col_time: np.arange(n)})
    
    df[config.col_interest_rate] = generate_interest_rate(config, rng)
    df[config.col_cheapest_flag] = generate_cheapest_flag(config, rng)
    
    # 2. Latent Variables
    trend = generate_trend(config, rng)
    seasonality = generate_seasonality(n, config.seasonality_amplitude_4w, config.seasonality_amplitude_52w)
    noise = rng.normal(0, config.noise_std, n)
    control_effect = df[config.col_interest_rate] * config.interest_rate_coef
    
    absolute_trend = config.base_sales + trend
    
    # Save latents to dataframe for counterfactual reconstruction later
    df['latent_trend'] = absolute_trend 
    df['latent_seasonality'] = seasonality
    df['latent_noise'] = noise
    df['latent_control_effect'] = control_effect
    
    # 3. Generate Spend, Transformations & Pricing Interactions
    transformed_data = {}
    sales_contribution_media = np.zeros(n)
    cheapest_mask = df[config.col_cheapest_flag] == 1
    
    for ch in config.channels:
        params = config.channel_params[ch]
        
        df[ch] = generate_spiky_spend(
            n, params["spend_prob"], params["spend_mean"], params["spend_std"], params["spend_min_clip"], rng
        )
        
        adstocked = geometric_adstock(df[ch].values, params["theta"])
        saturated = hill_function(adstocked, params["alpha"], params["gamma"])
        
        active_coef = np.where(cheapest_mask, params["coef_if_cheapest"], params["coef_if_not_cheapest"])
        contribution = saturated * active_coef
        
        transformed_data[ch] = {
            'adstocked': adstocked,
            'saturated_contribution': contribution
        }
        
        sales_contribution_media += contribution

    # 4. Assemble Target Variable
    df[config.col_target] = (
        absolute_trend + 
        seasonality +
        sales_contribution_media +
        control_effect +
        noise
    ).clip(0, None)
    
    transformed_data['latent'] = {
        'trend': absolute_trend,
        'seasonality': seasonality
    }
    
    return df, transformed_data


def calculate_true_sales(df: pd.DataFrame, config: MMMConfig) -> np.ndarray:
    """Calculates ground truth sales from given spend and saved latents."""
    sales_contrib = np.zeros(len(df))
    cheapest_mask = df[config.col_cheapest_flag] == 1

    for ch in config.channels:
        params = config.channel_params[ch]
        adstocked = geometric_adstock(df[ch].values, params["theta"])
        saturated = hill_function(adstocked, params["alpha"], params["gamma"])
        active_coef = np.where(cheapest_mask, params["coef_if_cheapest"], params["coef_if_not_cheapest"])
        sales_contrib += saturated * active_coef

    true_sales = (
        df['latent_trend'] + 
        df['latent_seasonality'] + 
        df['latent_control_effect'] + 
        df['latent_noise'] + 
        sales_contrib
    ).clip(0, None)
    
    return true_sales