import math
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def plot_mmm_features(df: pd.DataFrame,
                      channels: List[str],
                      true_transforms: Dict, 
                      price_per_unit: float,
                      estimated_transforms: Optional[Dict] = None,
                      time_col: str = 'week',
                      target_col: str = 'sales',
                      control_cols: Optional[List[str]] = None,
                      binary_cols: Optional[List[str]] = None,
                      trend_key: Optional[str] = 'trend',
                      save_path: Optional[str] = None):
    """
    Plots raw spend, controls, true latent effects, and estimated models 
    in a dynamic grid layout.
    """
    # 1. Increase global font sizes for readability
    plt.rcParams.update({
        'font.size': 12, 
        'axes.titlesize': 14, 
        'axes.labelsize': 12,
        'legend.fontsize': 10
    })

    # Default to our known controls if none are explicitly passed
    control_cols = control_cols or ['interest_rate']
    binary_cols = binary_cols or ['we_are_cheapest']

    # 2. Calculate Grid Layout dimensions
    # We plot: 1 (Sales) + 1 (Trend) + Controls + Binary + Channels
    n_plots = 2 + len(control_cols) + len(binary_cols) + len(channels)
    cols = 2
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * rows, 5 * rows))
    axes = axes.flatten() # Flatten so we can iterate linearly through a 1D array

    time_data = df[time_col]
    plot_idx = 0

# ---------------------------------------------------------
    # Plot 1: Target Variable (Sales)
    # ---------------------------------------------------------
    ax = axes[plot_idx]
    ax.plot(time_data, df[target_col], label=f'Total {target_col.title()} (Observed)', color='black', linewidth=2)
    ax.set_title(f"Observed {target_col.title()}")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    sec_ax = ax.secondary_yaxis('right', functions=(lambda x: x * price_per_unit, lambda x: x * price_per_unit))
    sec_ax.set_ylabel("Money Earned ($)")
    
    plot_idx += 1

    # ---------------------------------------------------------
    # Plot 2: Latent Trend (True vs Estimated)
    # ---------------------------------------------------------
    ax = axes[plot_idx]
    if trend_key and 'latent' in true_transforms and trend_key in true_transforms['latent']:
        ax.plot(time_data, true_transforms['latent'][trend_key], '--', label='True Trend', color='gray', linewidth=2)
        
        # Overlay Estimated Trend if provided
        if estimated_transforms and 'latent' in estimated_transforms and trend_key in estimated_transforms['latent']:
            ax.plot(time_data, estimated_transforms['latent'][trend_key], ':', 
                    label='Estimated Trend (MCMC)', color='orange', linewidth=3)
            
        ax.set_title("Underlying Trend")
        ax.legend(loc='upper left')
    else:
        ax.set_title("Underlying Trend (Not Found)")
    ax.grid(alpha=0.3)
    plot_idx += 1

    # ---------------------------------------------------------
    # Plot 3: Continuous Controls
    # ---------------------------------------------------------
    for ctrl in control_cols:
        ax = axes[plot_idx]
        ax.plot(time_data, df[ctrl], color='purple', linewidth=2, label=ctrl.replace('_', ' ').title())
        ax.set_title(f"Control Variable: {ctrl.replace('_', ' ').title()}")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        plot_idx += 1

    # ---------------------------------------------------------
    # Plot 4: Binary/Step Controls
    # ---------------------------------------------------------
    for bin_col in binary_cols:
        ax = axes[plot_idx]
        ax.step(time_data, df[bin_col], color='red', where='mid', linewidth=2, label=bin_col.replace('_', ' ').title())
        ax.set_yticks([0, 1])
        ax.set_title(f"Binary Flag: {bin_col.replace('_', ' ').title()}")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        plot_idx += 1

    # ---------------------------------------------------------
    # Plot 5: Channels (Spend vs Transformations)
    # ---------------------------------------------------------
    for ch in channels:
        ax = axes[plot_idx]
        ax2 = ax.twinx()
        
        # Raw Spend Bar Chart
        ax.bar(time_data, df[ch], alpha=0.3, color='blue', label=f'Raw {ch}')
        
        # True Transformation Line (FIXED)
        if true_transforms and ch in true_transforms:
            true_contrib = true_transforms[ch]['saturated_contribution']
            ax2.plot(time_data, true_contrib, color='green', linewidth=2, label='True Saturated Contribution')
        
        # Estimated Transformation Line & HDI Fill
        if estimated_transforms and ch in estimated_transforms:
            est_contrib = estimated_transforms[ch]['saturated_contribution']
            ax2.plot(time_data, est_contrib, color='orange', linestyle=':', 
                     linewidth=3, label='Estimated Contribution')
            
            if 'lower' in estimated_transforms[ch] and 'upper' in estimated_transforms[ch]:
                ax2.fill_between(time_data, 
                                 estimated_transforms[ch]['lower'], 
                                 estimated_transforms[ch]['upper'], 
                                 color='orange', alpha=0.2)

        ax.set_ylabel("Spend Amount ($)")
        ax2.set_ylabel("Sales Contribution")
        ax.set_title(f"Spend vs Contribution: {ch.replace('_', ' ').title()}")
        
        # Combine legends from both axes
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        if lines_1 or lines_2: # Only draw legend if there's something to show
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            
        ax.grid(alpha=0.3)
        plot_idx += 1

    # ---------------------------------------------------------
    # Clean up and Save/Show
    # ---------------------------------------------------------
    # If we have an odd number of plots, hide the empty final subplot
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()