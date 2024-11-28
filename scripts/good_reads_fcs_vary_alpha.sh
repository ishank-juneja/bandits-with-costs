#!/bin/bash

# Define path to the Python script
PYTHON_SCRIPT="src/param_variation_experiments/vary_alpha_fcs_setting.py"

# Common path variable
COMMON_PATH="data/bandit_instances"
# Result path variable
OUT_FILE_PATH="results/_run_logs/gr_fcs_vary_alpha/"

# Single file to use
FILE="${COMMON_PATH}/good_reads_instance.txt"

# Default parameters for the Python script
STEP=10000
HORIZON=5000000
NRUNS=25

echo "Running simulation on $FILE"

# Get number of CPU cores (assuming Linux)
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)

# Define algorithm array
ALGORITHMS=("cs-pe" "cs-etc" "cs-ucb" "cs-ts")

# Function to run process and echo completion
run_process() {
    local algo=$1
    local file=$2
    local step=$3
    local horizon=$4
    local nruns=$5
    local alpha=$6
    local core=$7
    local logname=$8
    local foldername=$9

    # Set process to specific core
    taskset -c "$core" python3 $PYTHON_SCRIPT -algos $algo -file "$file" -STEP $step -horizon $horizon -nruns $nruns -alpha $alpha > "${OUT_FILE_PATH}${foldername}/${logname}"
    echo "Completed: alpha = $alpha, algorithm = $algo, core = $core"
}

# Sweep over alpha values from 0.05 to 0.45 in increments of 0.05
index=0
for ALPHA in $(seq 0.05 0.05 0.45); do
	filename=$(basename -- "$FILE")
    foldername="${filename%.*}_alpha_${ALPHA}"
    mkdir -p "${OUT_FILE_PATH}${foldername}"
    for ALGO in "${ALGORITHMS[@]}"; do
        logname="${filename%.*}_alpha_${ALPHA}_${ALGO}_log.csv"
        echo "Running simulation for alpha = $ALPHA, algorithm = $ALGO on core $((index % NUM_CORES))"
        run_process $ALGO $FILE $STEP $HORIZON $NRUNS $ALPHA $((index % NUM_CORES)) $logname $foldername &
        ((index++))
    done
done

# Wait for all background processes to finish
wait

echo "All alpha simulations complete"

# Loop through each alpha value
for ALPHA in $(seq 0.05 0.05 0.45); do
    # Folder and file setup for concatenation
    filename=$(basename -- "$FILE")
    foldername="${filename%.*}_alpha_${ALPHA}"
    stitched_filename="${foldername}_log.csv"

    # Initialize the stitched file with the header of the first CSV file
    first_file="${OUT_FILE_PATH}${foldername}/$(ls ${OUT_FILE_PATH}${foldername} | head -n 1)"
    head -n 1 "$first_file" > "${OUT_FILE_PATH}${stitched_filename}"

    # Concatenate all CSV files for current alpha, skip headers after the first file
    for CSV_FILE in "${OUT_FILE_PATH}${foldername}"/*.csv; do
        tail -n +2 "$CSV_FILE" >> "${OUT_FILE_PATH}${stitched_filename}"
    done

    # Remove the individual algorithm files and the directory
    rm -r "${OUT_FILE_PATH}${foldername}"
done

echo "All files stitched and original directories cleaned up."

