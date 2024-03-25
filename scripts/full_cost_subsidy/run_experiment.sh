#!/bin/bash

# Common path variable
COMMON_PATH="data/bandit_instances/full_cost_subsidy"
# Result path variable
OUT_FILE_PATH="results/run_logs/full_cost_subsidy/"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/I1.txt"
)

# Default parameters for the Python script
STEP=100
HORIZON=400000
NRUNS=10

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies_fcs.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
done
