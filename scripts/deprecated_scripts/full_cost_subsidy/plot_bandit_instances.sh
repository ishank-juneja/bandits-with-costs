#!/bin/bash

# Common path variable
COMMON_PATH="data/bandit_instances/full_cost_subsidy"
RESULTS_PATH="data/bandit_instances/full_cost_subsidy"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/I1.txt"
    "${COMMON_PATH}/I2.txt"
    "${COMMON_PATH}/I3.txt"
    "${COMMON_PATH}/I4.txt"
    "${COMMON_PATH}/I5.txt"
)

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Plotting results for $file"
     # Plot quality regret
    python src/plotting/plot_fcs_instance.py --file "$file" --save-dir "$RESULTS_PATH"
done
