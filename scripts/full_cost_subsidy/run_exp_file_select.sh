#!/bin/bash

# Do some zenity stuff to select the file
# Prompt the user to select a file
file=$(zenity --file-selection --title="Select a file")

# Result path variable
# Prompt to select output directory
OUT_FILE_PATH=$(zenity --file-selection --directory --title="Select a directory to save the results")

# Default parameters for the Python script
STEP=500
HORIZON=1000000
NRUNS=50

echo "Running simulation on $file"

# Extract the basename of the file and append "_log"
filename=$(basename -- "$file")
logname="${filename%.*}_log.csv"

python src/simulate_play/simulate_policies_fcs.py -file "$file" -STEP $STEP -horizon $HORIZON -nruns $NRUNS > "${OUT_FILE_PATH}${logname}"
