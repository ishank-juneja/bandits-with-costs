#!/bin/bash

# Common path variable
COMMON_PATH="data/ml-25m"
# Result path variable
OUT_FILE_PATH="results/_run_logs/ml/"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/bandit_instance.txt"
)

# Default parameters for the Python script
STEP=500
HORIZON=5000000
NRUNS=100

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies_fcs.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
done

echo "All simulations complete"
