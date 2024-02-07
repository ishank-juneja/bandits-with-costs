#!/bin/bash

# Directory containing the problem instance files
INSTANCE_DIR="data/bandit_instances/mtr"

# Default parameters for the Python script
STEP=100
HORIZON=2000000
NRUNS=50

# Loop through each .txt file in the directory
for file in "$INSTANCE_DIR"/*.txt; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies.py -idx "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "results/run_logs/mtr/$logname"
done
