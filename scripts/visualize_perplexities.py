import argparse
import matplotlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="bigcode/santacoder")
    parser.add_argument("--dataset_name", type=str, default="codeparrot/github-code")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_languages", type=str, default="Python")
    parser.add_argument("--dataset_licenses", type=str, default="gpl-3.0")
    parser.add_argument("--dataset_size", type=int, default=-1)
    parser.add_argument("--dataset_max_length", type=int, default=4096)
    parser.add_argument("--dataset_output_dir", type=str, default="data/dataset")
    return parser.parse_args()

sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({'font.size': 28})

def visualize_perplexities(perplexities, tokens, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=perplexities, palette="Set2")
    ax.set_xticklabels(tokens)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(filename)
    plt.clf()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "cross_entropy.pkl", "rb") as f:
    cross_dist = pickle.load(f)

with open(output_dir / "token_cnt.pkl", "rb") as f:
    token_cnt = pickle.load(f)

most_common = token_cnt.most_common()
method_invocations = [
    t for t in most_common
    if t[0].startswith("<argument_list") or t[0].startswith("<call")
]
internals = [t for t in argument_lists if "internal" in t[0]]
externals = [t for t in argument_lists if "internal" not in t[0]]

internal_crosses = [
    cross_dist[token]
    for token, _ in internals
]
external_crosses = [
    cross_dist[token]
    for token, _ in externals
]

visualize_perplexities(
    internal_crosses,
    internals,
    "Cross Entropy of Argument Lists",
    "internal_invocations.png",
)
visualize_perplexities(
    external_crosses,
    externals,
    "Cross Entropy of Argument Lists",
    "external_invocations.png",
)