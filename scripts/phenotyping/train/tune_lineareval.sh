#!/bin/bash

# Define the range of values
values=(0.0429 0.0794 0.0341 0.0307 0.0415 0.0508 0.0095 0.0063 0.0577 0.0399)

# Loop through the values
for value in "${values[@]}"; do
    # Pass the value as a variable to another script
    sbatch ./lineareval.sh "$value"
done
