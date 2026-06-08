import os
from simulation.config import MMMConfig
from simulation.plotting import plot_mmm_features
from simulation.simulation import simulate_mmm_data

def main():
    # 1. Define output paths
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "mmm_simulated_data.csv")
    plot_path = os.path.join(output_dir, "mmm_simulation_plot.png")

    # 2. Initialize configuration (Template)
    print("Initializing simulation configuration...")
    config = MMMConfig()
    
    # Example: You can override parameters dynamically if needed here:
    # config.noise_std = 150.0 
    # config.trend_type = 'linear'

    # 3. Generate the dataset and extract the true underlying variables
    print("Running MMM data simulation...")
    df_observed, true_effects = simulate_mmm_data(config)

    # 4. Save the simulated dataset
    df_observed.to_csv(csv_path, index=False)
    print(f"Data successfully saved to: {csv_path}")

    # 5. Visualize the data and save the plot
    print("Generating and saving visualizations...")
    plot_mmm_features(
        df=df_observed, 
        channels=config.channels, 
        true_transforms=true_effects,
        time_col=config.col_time,
        target_col=config.col_target,
        save_path=plot_path
    )
    
    print("Simulation complete.")

if __name__ == "__main__":
    main()