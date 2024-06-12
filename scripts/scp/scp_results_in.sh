#!/bin/bash

# Remote server details
SSH_HOST="lions"
REMOTE_PATH="/home/ijuneja/bandits-with-costs/results/plots"

# Local directory where files will be copied
LOCAL_PATH="/home/ishank/PycharmProjects/bandits-with-costs/results/plots_lions_server/"

# SCP command to copy files
scp -r $SSH_HOST:$REMOTE_PATH $LOCAL_PATH

# Print completion message
echo "Files have been copied successfully."
