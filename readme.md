# MMM Simulation

### Concatenate all code in a folder
```bash
python concatenate_files.py -i simulation -o simulation.txt
```

## main.py commands venv

### Generating data

Take a config.json and generate a data .csv and a plot of the scenario

```bash
python main.py generate -c configurations/config.json -d output/my_scenario.csv -p output/my_scenario_plot.png
```

### Generating a plot based on a data .csv
Take a .csv and generate a plot of the scenario

```bash
python main.py plot-data -d output/my_scenario.csv -c configurations/config.json -p output/backup_data_plot.png
```

### Run a simulation and fit a model
```bash
python main.py analyze -d output/my_scenario.csv -c configurations/config.json -r configurations/good_priors.json -p output/parameter_recovery_plot.png -t output/trace.nc
```

### Create response curves
```bash
python main.py plot-response -d output/my_scenario.csv -c configurations/config.json -t output/trace.nc -p output/response_curve_plot.png
```

## main.py command podman

### Generating data with podman-compose
```bash
podman-compose exec meridian-env python main.py generate -c configurations/simple_config.json -d output/my_scenario.csv -p output/my_scenario_plot.png
```

### Generating a meridian analysis
```bash
podman-compose exec meridian-env python main.py meridian-analyze -c configurations/simple_config.json -d output/my_scenario.csv -p output/meridian_eval
```