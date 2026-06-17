import numpy as np

def geometric_adstock(spend: np.ndarray, theta: float) -> np.ndarray:
    """
    Applies geometric decay (adstock) to a spend array.
    
    Args:
        spend: Array of spend values per time step.
        theta: Decay rate parameter (0 <= theta < 1).
        
    Returns:
        Array of adstocked spend values.
    """
    adstocked = np.zeros_like(spend, dtype=float)
    if len(spend) > 0:
        adstocked[0] = spend[0]
        for t in range(1, len(spend)):
            adstocked[t] = spend[t] + theta * adstocked[t-1]
    return adstocked

def hill_function(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Applies the Hill function for saturation.
    
    Args:
        x: Array of input values (typically adstocked spend).
        alpha: Shape parameter (controls the 'S' shape steepness).
        gamma: Half-saturation point.
        
    Returns:
        Saturated values (between 0 and 1).
    """
    # Add epsilon to prevent division by zero
    epsilon = 1e-8
    x_safe = np.clip(x, epsilon, None)
    return 1.0 / (1.0 + (gamma / x_safe) ** alpha)