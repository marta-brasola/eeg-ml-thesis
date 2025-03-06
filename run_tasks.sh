#!/bin/bash

# Define an array of task names
tasks=("A_vs_C" "A_vs_F" "F_vs_C" "A_vs_F_vs_C")

# Loop through each task and execute the Python script
for task in "${tasks[@]}"; do
    echo "Running training script with task: $task"
    python3 milt_training.py --task "$task" 
    echo "Finished task: $task"
    echo "--------------------------------"
done
