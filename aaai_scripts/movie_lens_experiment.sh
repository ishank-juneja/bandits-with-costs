#!/bin/bash

# Define path to the Python script
PYTHON_SCRIPT="src/ICLR_rebuttal_experiments/new_movie_lens.py"

# Common path variable
COMMON_PATH="aaai_data/bandit_instances"
# Result path variable
OUT_FILE_PATH="aaai_results/_run_logs/ml/"

# Single file to use
FILE="${COMMON_PATH}/movie_lens_instance.txt"

# Default parameters for the Python script
STEP=1000
HORIZON=5000000
NRUNS=50

# Notify about the simulation being run
echo "Running simulation on $FILE"

# Get number of CPU cores (assuming Linux)
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)

# Sweep over alpha values from 0.05 to 0.45 in increments of 0.05
index=0
for ALPHA in $(seq 0.05 0.05 0.45); do
    # Extract the basename of the file and format logname with the current alpha value
    filename=$(basename -- "$FILE")
    logname="${filename%.*}_alpha_${ALPHA}_log.csv"

    # Using taskset to set the Python process to a specific core
    # Using modulo to cycle through cores if more tasks than cores
    echo "Running simulation for alpha = $ALPHA on core $((index % NUM_CORES))"
    taskset -c $((index % NUM_CORES)) python3 $PYTHON_SCRIPT -file "$FILE" -STEP $STEP -horizon $HORIZON -nruns $NRUNS -alpha $ALPHA > "${OUT_FILE_PATH}${logname}" &

    # Increment the index to cycle through the CPU cores
    ((index++))
done

# Wait for all background processes to finish
wait

echo "All alpha simulations complete"
