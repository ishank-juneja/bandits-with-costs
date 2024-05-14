#!/bin/bash

# Specify the directory where you want to save the files.
output_dir="data/bandit_instances/full_cost_subsidy/computer_generated"


# Make sure the output directory exists.
mkdir -p "$output_dir"

# Initialize a counter for the file index.
file_idx=1

# Generate 11 linearly spaced values between 0.3 and 0.6.
for x in $(seq 0.4 0.05 0.9); do
    # Format the file name.
    file_name=$(printf "I%03d.txt" "$file_idx")
    full_path="$output_dir/$file_name"

    # Prepare the content with the current value of x.
    content="instance_id: FCS%03d\narm_reward_array: 0.65, $x, 0.83, 0.95\nsubsidy_factor: 0.2\narm_cost_array: 0.75, 0.8, 0.85, 0.9"
    content=$(printf "$content" "$file_idx")

    # Write the content to the file.
    echo -e "$content" > "$full_path"

    # Increment the file index for the next iteration.
    ((file_idx++))
done
