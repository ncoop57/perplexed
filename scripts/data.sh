#! /bin/bash

python data_preparation.py \
    --model_name bigcode/santacoder \
    --dataset_name codeparrot/github-code \
    --dataset_split train \
    --dataset_percent 0.1 \
    --dataset_languages Python \
    --dataset_licenses gpl-3.0 \
    --dataset_max_length 4096 \
    --dataset_output_dir /home/toolkit/perplexed_data/dataset_random_10pct \
    --num_proc 96