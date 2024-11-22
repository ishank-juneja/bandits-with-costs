#!/bin/bash

# Define path to the Python script
PYTHON_SCRIPT="src/ICLR_rebuttal_experiments/new_movie_lens_known_ell.py"

# Common path variable
COMMON_PATH="aaai_data/bandit_instances"
# Result path variable
OUT_FILE_PATH="aaai_results/_run_logs/ml_known_ell/"

# Single file to use
FILE="${COMMON_PATH}/movie_lens_instance.txt"

# Default parameters for the Python script
STEP=1000
HORIZON=5000000
NRUNS=2

echo "Running simulation on $FILE"

# Get number of CPU cores (assuming Linux)
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)

# Define algorithm array
ALGORITHMS=("pe" "asymmetric-pe" "ucb-cs")

# Function to run process and echo completion
run_process() {
    local algo=$1
    local file=$2
    local step=$3
    local horizon=$4
    local nruns=$5
    local ell=$6
    local core=$7
    local logname=$8
    local foldername=$9

    # Set process to specific core
    taskset -c "$core" python3 $PYTHON_SCRIPT -algos $algo -file "$file" -STEP $step -horizon $horizon -nruns $nruns -ell $ell > "${OUT_FILE_PATH}${foldername}/${logname}"
    echo "Completed: ell = ell, algorithm = $algo, core = $core"
}

# Sweep over ell values from 0 - 19 in intervals of 4
index=0
for ELL in $(seq 0 4 19); do
    filename=$(basename -- "$FILE")
    foldername="${filename%.*}_ell_${ELL}"
    mkdir -p "${OUT_FILE_PATH}${foldername}"
    for ALGO in "${ALGORITHMS[@]}"; do
        logname="${filename%.*}_ell_${ELL}_${ALGO}_log.csv"
        echo "Running simulation for ell = $ELL, algorithm = $ALGO on core $((index % NUM_CORES))"
        run_process $ALGO $FILE $STEP $HORIZON $NRUNS $ELL $((index % NUM_CORES)) $logname $foldername &
        ((index++))
    done
done

# Wait for all background processes to finish
wait

echo "All ell simulations complete"

# Loop through each ell value
for ELL in $(seq 0 4 19); do
    # Folder and file setup for concatenation
    filename=$(basename -- "$FILE")
    foldername="${filename%.*}_ell_${ELL}"
    stitched_filename="${foldername}_log.csv"

    # Initialize the stitched file with the header of the first CSV file
    first_file="${OUT_FILE_PATH}${foldername}/$(ls ${OUT_FILE_PATH}${foldername} | head -n 1)"
    head -n 1 "$first_file" > "${OUT_FILE_PATH}${stitched_filename}"

    # Concatenate all CSV files for current ell, skip headers after the first file
    for CSV_FILE in "${OUT_FILE_PATH}${foldername}"/*.csv; do
        tail -n +2 "$CSV_FILE" >> "${OUT_FILE_PATH}${stitched_filename}"
    done

    # Remove the individual algorithm files and the directory
    rm -r "${OUT_FILE_PATH}${foldername}"
done

echo "All files stitched and original directories cleaned up."
