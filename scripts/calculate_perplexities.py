import argparse
import datasets
import pickle

from code_tokenizers.core import CodeTokenizer
from datasets import load_from_disk
from functools import partial
from pathlib import Path
from perplexed.core import perplexed
from transformers import AutoModelForCausalLM, default_data_collator

datasets.logging.set_verbosity(datasets.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/dataset")
    parser.add_argument("--dataset_column", type=str, default="code")
    parser.add_argument("--semantic_column", type=str, default="merged_ast")
    parser.add_argument("--model_name", type=str, default="bigcode/santacoder")
    parser.add_argument("--tokenizer_language", type=str, default="python")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="data/perplexities")
    return parser.parse_args()

def code_collator(batch):
    merged_ast = []
    for b in batch:
        merged_ast.append(b.pop(args.semantic_column))
    batch = default_data_collator(batch)
    batch[args.semantic_column] = merged_ast
    return batch

def tokenizer_wrapper(tokenizer, example, column, *args, **kwargs):
    return tokenizer(example[column], internal_methods=example["internal_methods"], *args, **kwargs)

args = parse_args()

# Load the dataset
ds = load_from_disk(args.dataset_path)

# Setup tokenizer
py_tokenizer = CodeTokenizer.from_pretrained(args.model_name, args.tokenizer_language, padding_token="<|endoftext|>")
tokenizer = partial(tokenizer_wrapper, py_tokenizer, column=args.dataset_column)
tokenizer.decode = py_tokenizer.decode

model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
model = model.to(args.device)
cross_dist, token_cnt = perplexed(
    model,
    ds,
    tokenizer=tokenizer,
    column=args.dataset_column,
    semantic_column=args.semantic_column,
    batch_size=args.batch_size,
    num_proc=args.num_proc,
    device=args.device,
    collate_fn=code_collator,
    pass_row=True,
    return_tokens=True,
    return_distributions=True,
    compute_perplexity=False,
)

# Save the cross entropy and token count
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "cross_entropy.pkl", "wb") as f:
    pickle.dump(cross_dist, f)

with open(output_dir / "token_cnt.pkl", "wb") as f:
    pickle.dump(token_cnt, f)