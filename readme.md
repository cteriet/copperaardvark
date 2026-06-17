# MMM Simulation

## main.py commands

### Generating data

Take a config.json and generate a data .csv and a plot of the scenario

```bash
python main.py generate -c configurations/config.json -d my_scenario.csv -p my_scenario_plot.png
```

### Generating a plot based on a data .csv
Take a .csv and generate a plot of the scenario

```bash
python main.py plot-data -d output/my_scenario.csv -c configurations/config.json -p output/backup_data_plot.png
```

### Run a simulation and fit a model

```bash
python main.py analyze -d output/my_scenario.csv -c configurations/config.json -p output/parameter_recovery_plot.png -t output/trace.nc
```

### Concatenate all code in a folder
```bash
python concatenate_files.py -i simulation -o simulation.txt
```

### Create response curves
```bash
python main.py plot-response -d output/my_scenario.csv -c configurations/config.json -t output/trace.nc -p output/response_curve_plot.png
```