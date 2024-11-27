#!/bin/bash

# Common path variable
COMMON_PATH="aaai_data/bandit_instances/toy2/"
# Result path variable
OUT_FILE_PATH="aaai_results/_run_logs/toy2/"

# Default parameters for the Python script
STEP=1000
HORIZON=500000
NRUNS=50

# Use find with a while read loop to process each file
find "${COMMON_PATH}" -name "*.txt" | while read file; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies_fcs.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
done
