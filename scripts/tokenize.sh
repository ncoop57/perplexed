#! /bin/bash

python tokenize_dataset.py \
    --dataset_path /home/toolkit/perplexed_data/dataset_random_10pct \
    --dataset_column code \
    --semantic_column merged_ast \
    --model_name bigcode/santacoder \
    --tokenizer_language python \
    --device cuda \
    --batch_size 32 \
    --num_proc 48 \
    --dataset_output_dir /home/toolkit/perplexed_data/tokenized_dataset_random_10pct