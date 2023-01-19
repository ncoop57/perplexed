#! /bin/bash

python calculate_perplexities.py \
    --dataset_path /home/toolkit/perplexed_data/dataset \
    --dataset_column code \
    --semantic_column merged_ast \
    --model_name bigcode/santacoder \
    --tokenizer_language python \
    --device cuda \
    --batch_size 32 \
    --num_proc 64 \
    --output_dir /home/toolkit/perplexed_data/perplexities