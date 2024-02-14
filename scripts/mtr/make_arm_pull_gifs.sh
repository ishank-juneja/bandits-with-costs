#!/bin/bash

# Directory containing the problem instance files
LOG_DIR="results/run_logs"

# Default parameters for the Python script
STEP=20000
HORIZON=2000000

# Loop through each .txt file in the directory
for file in "$LOG_DIR"/*.csv; do
    echo "Plotting results for file $file"

    python src/plotting/arms_distribution.py -idx "$file" -STEP $STEP -horizon $HORIZON
done
