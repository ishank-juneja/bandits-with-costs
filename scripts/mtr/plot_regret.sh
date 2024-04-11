#!/bin/bash

# Common path variable
COMMON_PATH="results/_run_logs/mtr"
RESULTS_PATH="results/plots/mtr"

# Variables for algos and metric
ALGOS="ucb mtr-ucb"

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
    python src/plotting/plotter.py --log-file "$file" --algos $ALGOS --metric qual_reg --save-dir "$RESULTS_PATH"
    # Plot cost regret
    python src/plotting/plotter.py --log-file "$file" --algos $ALGOS --metric cost_reg --save-dir "$RESULTS_PATH"
done
