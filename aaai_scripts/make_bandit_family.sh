#!/bin/bash

# Specify the directory where you want to save the files.
output_dir="aaai_data/bandit_instances/toy2"

# Make sure the output directory exists.
mkdir -p "$output_dir"

# Initialize a counter for the file index.
file_idx=1

# Generate 11 linearly spaced values between 0.3 and 0.6.
for x in $(seq 0.78 0.04 0.86); do
    # Format the file name.
    file_name=$(printf "I%03d.txt" "$file_idx")
    full_path="$output_dir/$file_name"

    # Prepare the content with the current value of x.
    content="instance_id: LCR%03d\narm_reward_array: $x, 0.9\nsubsidy_factor: 0.1\narm_cost_array: 0.1, 0.5"
    content=$(printf "$content" "$file_idx")

    # Write the content to the file.
    echo -e "$content" > "$full_path"

    # Increment the file index for the next iteration.
    ((file_idx++))
done
