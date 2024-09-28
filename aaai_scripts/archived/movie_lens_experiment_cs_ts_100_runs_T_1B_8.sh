#!/bin/bash

# Common path variable
COMMON_PATH="aaai_data/bandit_instances"
# Result path variable
OUT_FILE_PATH="aaai_results/_run_logs/ml/"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/movie_lens_instance.txt"
)

# Default parameters for the Python script
STEP=1000000
HORIZON=1000000000

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_cs_ts_100_runs_1B_8_log.csv"

    python3 src/simulate_play/simulate_ts_cs_71_80.py -file "$file" -STEP $STEP -horizon $HORIZON > "${OUT_FILE_PATH}${logname}"
done

echo "All simulations complete"
