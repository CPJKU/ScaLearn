#!/bin/bash

# Array of target_task values
DATASET_NAME=("cb" "copa" "boolq" "wic" "wsc" "rte" "mrpc" "mnli" "qnli" "sst2" "stsb" "qqp" "multirc" "cola" "record")

# Array of OMEGA values
OMEGA=("00" "01" "03" "05" "07" "09" "1")

# Set RUN_NAME and SEED
RUN_NAME="st-a-3e-4"
SEEDS=(0 1 2 3 4)

# Loop through target_task and OMEGA
for source_task in "${DATASET_NAME[@]}"
do
  for target_task in "${DATASET_NAME[@]}"
   do
        for omega in "${OMEGA[@]}"
        do
          for SEED in "${SEEDS[@]}"
          do    
            # these tasks only run with seeds 0 to 4
            if [ $SEED -gt 2 ] && [ $target_task = "boolq" -o $target_task = "stsb" -o $target_task = "wic" -o $target_task = "cola" ]; then
                echo "Skipping $target_task with seed $SEED"
                continue
            fi

            if [ $SEED -gt 0 ] && [ $target_task = "mnli" -o $target_task = "qqp" -o $target_task = "qnli" -o $target_task = "sst2" -o $target_task = "record" -o $target_task = "multirc" ]; then
                echo "Skipping $target_task with seed $SEED"
                continue
            fi

            # Create the target directory path
            target_dir="../../runs/PPROBE/$RUN_NAME/$omega/$source_task/$target_task/roberta-base/100/"
            
            # Create the target directory if it doesn't exist
            mkdir -p "$target_dir"
            
            # Copy the lowest subdirectory to the target directory
            cp -r "$RUN_NAME/$source_task/roberta-base/100/$SEED/" "$target_dir"
            done
        done
    done
done
