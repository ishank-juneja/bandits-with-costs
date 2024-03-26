#!/bin/bash

# Common path variable
COMMON_PATH="data/bandit_instances/full_cost_subsidy/computer_generated/"
# Result path variable
OUT_FILE_PATH="results/run_logs/full_cost_subsidy/computer_generated/"

# Default parameters for the Python script
STEP=1
HORIZON=5000
NRUNS=10

# Use find with a while read loop to process each file
find "${COMMON_PATH}" -name "*.txt" | while read file; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies_fcs.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
done
