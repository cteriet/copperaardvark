import numpy as np
from simulation.config import MMMConfig

def generate_gp(n_weeks: int, length_scale: float, variance: float, rng: np.random.Generator) -> np.ndarray:
    """Core helper function to generate a Gaussian Process."""
    x = np.arange(n_weeks)[:, None]
    K = np.exp(-0.5 * (x - x.T)**2 / (length_scale**2)) * variance
    # Add numerical jitter to the diagonal to ensure matrix is positive semi-definite
    K += 1e-6 * np.eye(n_weeks) 
    return rng.multivariate_normal(np.zeros(n_weeks), K)

def generate_interest_rate(config: MMMConfig, rng: np.random.Generator) -> np.ndarray:
    """Generates the interest rate variable based on the requested dynamics."""
    if config.interest_rate_type == 'independent':
        rates = rng.normal(config.interest_rate_mean, config.interest_rate_std, config.n_weeks)
    elif config.interest_rate_type == 'sticky':
        gp = generate_gp(config.n_weeks, config.interest_rate_gp_length_scale, config.interest_rate_gp_variance, rng)
        rates = config.interest_rate_mean + gp
    else:
        raise ValueError(f"Unknown interest_rate_type: {config.interest_rate_type}")
    
    return np.clip(rates, config.interest_rate_min_clip, None)

def generate_cheapest_flag(config: MMMConfig, rng: np.random.Generator) -> np.ndarray:
    """Generates the boolean pricing flag based on the requested dynamics."""
    if config.cheapest_flag_type == 'independent':
        return rng.binomial(1, config.we_are_cheapest_prob, config.n_weeks)
    elif config.cheapest_flag_type == 'sticky':
        # Generate a standard GP
        gp = generate_gp(config.n_weeks, config.cheapest_flag_gp_length_scale, 1.0, rng)
        # Find the value at which exactly `we_are_cheapest_prob` % of the data is above the threshold
        threshold = np.percentile(gp, 100 * (1 - config.we_are_cheapest_prob))
        return (gp > threshold).astype(int)
    else:
        raise ValueError(f"Unknown cheapest_flag_type: {config.cheapest_flag_type}")

def generate_spiky_spend(n_weeks: int, prob: float, mean: float, std: float, min_clip: float, rng: np.random.Generator) -> np.ndarray:
    """Generates clipped normal distributed marketing expenses at random intervals."""
    spikes = rng.binomial(1, prob, n_weeks)
    amounts = rng.normal(mean, std, n_weeks)
    amounts = np.clip(amounts, min_clip, None) 
    return spikes * amounts

def generate_trend(config: MMMConfig, rng: np.random.Generator) -> np.ndarray:
    """Generates the trend variable based on the requested dynamics."""
    n_weeks = config.n_weeks
    x = np.arange(n_weeks)
    
    if config.trend_type == 'zero':
        return np.zeros(n_weeks)
    elif config.trend_type == 'linear':
        return x * config.trend_linear_slope
    elif config.trend_type == 'spline':
        y = np.zeros(n_weeks)
        for knot_prop, slope in zip(config.trend_spline_knots, config.trend_spline_slopes):
            knot_idx = int(n_weeks * knot_prop)
            y[knot_idx:] += (x[knot_idx:] - knot_idx) * slope
        return y
    elif config.trend_type == 'gp':
        return generate_gp(n_weeks, config.trend_gp_length_scale, config.trend_gp_variance, rng)
    else:
        raise ValueError(f"Unknown trend_type: {config.trend_type}")

def generate_seasonality(n_weeks: int, amp_4w: float, amp_52w: float) -> np.ndarray:
    """Generate a 4 week and 52 week seasonality trend."""
    t = np.arange(n_weeks)
    season_4w = amp_4w * np.sin(2 * np.pi * t / 4)
    season_52w = amp_52w * np.sin(2 * np.pi * t / 52)
    return season_4w + season_52w