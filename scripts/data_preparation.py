import argparse
import datasets
import logging
import random

from code_tokenizers.core import CodeTokenizer
from code_tokenizers.helpers import get_internal_methods
from datasets import load_dataset
from rich.logging import RichHandler
from rich.progress import track

datasets.logging.set_verbosity(datasets.logging.ERROR)
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")

# set seed
random.seed(115)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bigcode/santacoder")
    parser.add_argument("--dataset_name", type=str, default="codeparrot/github-code")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_percent", type=float, default=1.)
    parser.add_argument("--dataset_languages", type=str, default="Python")
    parser.add_argument("--dataset_licenses", type=str, default="gpl-3.0")
    parser.add_argument("--dataset_max_length", type=int, default=4096)
    parser.add_argument("--dataset_output_dir", type=str, default="data/dataset")
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()

def find_duplicates(items):
  # Create an empty set to store the items that we have already seen
  seen = set()

  # Create an empty list to store the duplicates that we find
  duplicates = []

  # Loop through each item in the list
  for item in items:
    # If the item is already in the "seen" set, then it must be a duplicate
    if item in seen:
      # Add the duplicate to the list
      duplicates.append(item)
    # If the item is not in the "seen" set, then add it to the set
    else:
      seen.add(item)

  # Return the list of duplicates
  return duplicates

def filter_repos(ds):
  datasets.disable_progress_bar()
  repo_names = find_duplicates(ds["repo_name"])

  repo_files = {}
  for repo_name in track(repo_names, description="Filtering repos"):
      rows_w_repo = ds.filter(
        lambda example: example["repo_name"] == repo_name,
        num_proc=4
      )

      if len(rows_w_repo) > 1:
          repo_files[repo_name] = [row["code"] for row in rows_w_repo]
  
  datasets.enable_progress_bar()

  # filter out repos with only one file
  ds = ds.filter(lambda example: example["repo_name"] in repo_files, num_proc=args.num_proc)
  return ds, repo_files

args = parse_args()

ds = load_dataset(
  args.dataset_name,
  split=args.dataset_split, #f"{args.dataset_split}[:{args.dataset_percent}%]",
  streaming=False,
  languages=[args.dataset_languages],
  licenses=[args.dataset_licenses],
)
# log the dataset size
log.info(f"Dataset size: {len(ds)}")

if args.dataset_percent < 1.:
  sz = len(ds)
  new_sz = int(args.dataset_percent * sz)
  ds = ds.select(random.sample(range(sz), new_sz))

  # log the dataset size
  log.info(f"Dataset size after sampling: {len(ds)}")

filtered_ds = ds.filter(lambda example: len(example["code"]) < args.dataset_max_length, num_proc=args.num_proc)
log.info(f"Dataset size after filtering by max length: {len(filtered_ds)}")
filtered_ds, repo_files = filter_repos(filtered_ds)
log.info(f"Dataset size after filtering by repos: {len(filtered_ds)}")

py_tokenizer = CodeTokenizer.from_pretrained(args.model_name, "python", padding_token="<|endoftext|>")

def example_internal_methods(example):
    internal_methods = get_internal_methods(
        [example["code"]],
        py_tokenizer
    )
    repo_internal_methods = get_internal_methods(
        repo_files[example["repo_name"]],
        py_tokenizer
    )
    repo_internal_methods = repo_internal_methods - internal_methods
    return repo_internal_methods

# add the internal methods to the dataset
filtered_ds = filtered_ds.map(
    lambda example: {
        "internal_methods": example_internal_methods(example)
    },
    num_proc=args.num_proc
)
# filter out repos with no internal methods
filtered_ds = filtered_ds.filter(lambda example: len(example["internal_methods"]) > 0, num_proc=args.num_proc)
log.info(f"Dataset size after filtering by internal methods: {len(filtered_ds)}")

# save the dataset
filtered_ds.save_to_disk(args.dataset_output_dir)