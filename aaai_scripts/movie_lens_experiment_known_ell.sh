#!/bin/bash

# Common path variable
COMMON_PATH="aaai_data/bandit_instances"
# Result path variable
OUT_FILE_PATH="aaai_results/_run_logs/ml/"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/movie_lens_instance_ell_10.txt"
)

# Default parameters for the Python script
STEP=1000
HORIZON=5000000
NRUNS=100

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python3 src/simulate_play/simulate_policies_ref_ell_setting.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
done

echo "All simulations complete"
