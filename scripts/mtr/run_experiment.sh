#!/bin/bash

# Common path variable
COMMON_PATH="data/bandit_instances/mtr"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/I1.txt"
#    "${COMMON_PATH}/I2.txt"
#    "${COMMON_PATH}/I3.txt"
#    "${COMMON_PATH}/I4.txt"
#    "${COMMON_PATH}/I5.txt"
)

# Default parameters for the Python script
STEP=100
HORIZON=200000
NRUNS=2

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Running simulation on $file"

    # Extract the basename of the file and append "_log"
    filename=$(basename -- "$file")
    logname="${filename%.*}_log.csv"

    python src/simulate_play/simulate_policies_mtr_setting.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "results/run_logs/mtr/$logname"
done
