from dataclasses import dataclass, field
from typing import Literal, Dict, List

@dataclass
class MMMConfig:
    """Configuration object for MMM Data Simulation."""
    n_weeks: int = 104
    base_sales: float = 1000.0
    random_seed: int = 42
    price_per_unit: float = 10.0
    
    # Column Names
    col_time: str = 'week'
    col_target: str = 'sales'

    # Controls for interest rate variable
    col_interest_rate: str = 'interest_rate'
    interest_rate_type: Literal['independent', 'sticky'] = 'sticky'
    interest_rate_mean: float = 3.5
    interest_rate_std: float = 0.5  
    interest_rate_gp_length_scale: float = 26.0  
    interest_rate_gp_variance: float = 0.25 
    interest_rate_min_clip: float = 0.0
    interest_rate_coef: float = -50.0
    
    # --- Pricing Flag Parameters ---
    col_cheapest_flag: str = 'we_are_cheapest'
    cheapest_flag_type: Literal['independent', 'sticky'] = 'sticky'
    we_are_cheapest_prob: float = 0.4
    cheapest_flag_gp_length_scale: float = 12.0 
    
    # Unified Channels
    channels: List[str] = field(default_factory=lambda: [
        "tv_spend", "digital_spend", "radio_spend", "comp_tv_spend"
    ])
    
    # Explicit state-dependent coefficients per channel
    channel_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "tv_spend": {
            "spend_prob": 0.2, "spend_mean": 5000.0, "spend_std": 2000.0, "spend_min_clip": 100.0,
            "theta": 0.7, "alpha": 2.0, "gamma": 5000.0, 
            "coef_if_cheapest": 1.0, "coef_if_not_cheapest": 0.6  # High penalty if not cheapest
        },
        "digital_spend": {
            "spend_prob": 0.8, "spend_mean": 2000.0, "spend_std": 800.0, "spend_min_clip": 50.0,
            "theta": 0.3, "alpha": 1.5, "gamma": 2000.0, 
            "coef_if_cheapest": 1.5, "coef_if_not_cheapest": 1.0  # Moderate penalty
        },
        "radio_spend": {
            "spend_prob": 0.4, "spend_mean": 1500.0, "spend_std": 500.0, "spend_min_clip": 0.0,
            "theta": 0.5, "alpha": 1.8, "gamma": 1000.0, 
            "coef_if_cheapest": 0.5, "coef_if_not_cheapest": 0.5  # No interaction effect!
        },
        "comp_tv_spend": {
            "spend_prob": 0.3, "spend_mean": 6000.0, "spend_std": 2500.0, "spend_min_clip": 200.0,
            "theta": 0.6, "alpha": 2.0, "gamma": 6000.0, 
            "coef_if_cheapest": 0.1, "coef_if_not_cheapest": -0.3 # Halo vs Cannibalization
        }
    })
    
    # Latent components
    trend_type: Literal['zero', 'linear', 'spline', 'gp'] = 'gp'
    trend_linear_slope: float = 5.0
    trend_spline_knots: List[float] = field(default_factory=lambda: [1/3, 2/3])
    trend_spline_slopes: List[float] = field(default_factory=lambda: [10.0, -15.0])
    trend_gp_length_scale: float = 20.0
    trend_gp_variance: float = 50000.0 
    
    seasonality_amplitude_4w: float = 50.0
    seasonality_amplitude_52w: float = 200.0

    noise_std: float = 100.0