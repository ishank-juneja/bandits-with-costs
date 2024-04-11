#!/bin/bash

# Common path variable
COMMON_PATH="results/_run_logs/no_cost_subsidy"
RESULTS_PATH="results/plots/no_cost_subsidy"


# Variables for algos
ALGOS="ucb improved-ucb"

# Hardcoded list of files using the common path variable
FILES=(
    "${COMMON_PATH}/I1_log.csv"
#    "${COMMON_PATH}/I2_log.csv"
#    "${COMMON_PATH}/I3_log.csv"
#    "${COMMON_PATH}/I4_log.csv"
#    "${COMMON_PATH}/I5_log.csv"
)

# Loop through each file in the hardcoded list
for file in "${FILES[@]}"; do
    echo "Plotting results for $file"
    # Plot quality regret
    python src/plotting/plotter.py --log-file "$file" --algos $ALGOS --metric reg --save-dir "$RESULTS_PATH"
done
