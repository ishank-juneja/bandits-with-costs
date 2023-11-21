#!/bin/bash

# Directory containing the problem instance files
LOG_DIR="../results/run_logs"

# Default parameters for the Python script
STEP=1
HORIZON=50000

# Loop through each .txt file in the directory
for file in "$INSTANCE_DIR"/*.txt; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python ../src/plotter.py -idx "$file" -STEP $STEP -horizon $HORIZON
done
