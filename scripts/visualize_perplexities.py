import argparse
import matplotlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import Counter
from datasets import load_from_disk
from pathlib import Path
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from wordcloud import WordCloud, STOPWORDS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenized_dataset_dir", type=str)
    return parser.parse_args()

sns.set_theme(style="whitegrid")

def visualize_perplexities(perplexities, tokens, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.boxplot(data=perplexities, palette="Set2")
    ax.set_xticklabels(tokens)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def visualize_bar(count, title, filename, avg=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [t[0] for t in count]
    y = [t[1] for t in count]
    ax = sns.barplot(x=x, y=y, palette="Set2")
    # draw the line for the average
    if avg is not None:
        ax.axhline(avg, ls="--", color="red")
        ax.text(0, avg + 0.1, f"Average: {avg:.3f}", color="red")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def visualize_word_cloud(tokens, filename):
    tokens = " ".join(tokens)
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = set(STOPWORDS),
                min_font_size = 10).generate(tokens)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)

args = parse_args()

root_dir = Path(args.root_dir)
root_dir.mkdir(parents=True, exist_ok=True)
with open(root_dir / "cross_entropy.pkl", "rb") as f:
    cross_dist = pickle.load(f)

with open(root_dir / "token_cnt.pkl", "rb") as f:
    token_cnt = pickle.load(f)

# remove endoftext token
# print(token_cnt.most_common(10))
endoftext = token_cnt.pop("<|endoftext|>")
print(f"Removed {endoftext} endoftext tokens")
print(len(token_cnt))
# filter out strings with non-ascci characters
token_cnt = Counter({
    token: cnt
    for token, cnt in token_cnt.items()
    if all(ord(c) < 128 for c in token) and "\x13" not in token
})
cross_dist = dict({
    token: cross
    for token, cross in cross_dist.items()
    if all(ord(c) < 128 for c in token) and "\x13" not in token
})
print(len(token_cnt))

most_common = token_cnt.most_common()
method_invocations = [
    t for t in most_common
    if t[0].startswith("<argument_list") or t[0].startswith("<call")
]
internals = [t for t in method_invocations if "internal" in t[0]]
externals = [t for t in method_invocations if "internal" not in t[0]]
externals = [t for t in externals if "call -> float" not in t[0] and "call -> string" not in t[0] and "call -> comment" not in t[0]]
internals = sorted(internals, key=lambda x: x[0])
externals = sorted(externals, key=lambda x: x[0])

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
    "Cross Entropy of Internal Method Invocations",
    root_dir / "internal_invocations.png",
)
visualize_perplexities(
    external_crosses,
    externals,
    "Cross Entropy of External Method Invocations",
    root_dir / "external_invocations.png",
)

# Write out the results to a table
table = Table(show_header=True, header_style="bold magenta", title="Internal vs External Method Invocations")
table.add_column("AST Node", style="dim", width=30)
table.add_column("Count", justify="right")
table.add_column("Internal?", justify="right")
table.add_column("Avg. Cross Entropy", justify="right")
table.add_column("Std. Cross Entropy", justify="right")

# interweave the internal and external method invocations
for (in_token, in_cnt), (ex_token, ex_cnt) in zip(internals, externals):
    # use ex token here to make it look better. Make sure to stay to 3 decimal places
    table.add_row(ex_token, str(in_cnt), "Yes", f"{np.mean(cross_dist[in_token]):.3f}", f"{np.std(cross_dist[in_token]):.3f}")
    table.add_row(ex_token, str(ex_cnt), "No", f"{np.mean(cross_dist[ex_token]):.3f}", f"{np.std(cross_dist[ex_token]):.3f}")

# Add summary row
table.add_row("Summary", "", "", "")
table.add_row("Internal", str(sum([t[1] for t in internals])), "", f"{np.mean([np.mean(cross_dist[t[0]]) for t in internals]):.3f}", f"{np.std([np.mean(cross_dist[t[0]]) for t in internals]):.3f}")
table.add_row("External", str(sum([t[1] for t in externals])), "", f"{np.mean([np.mean(cross_dist[t[0]]) for t in externals]):.3f}", f"{np.std([np.mean(cross_dist[t[0]]) for t in externals]):.3f}")

console = Console()
console.print(table)

ast_tokens = [
    t for t in most_common
    if (t[0].startswith("<") or t[0].endswith(">")) and "->" in t[0]
][:10]
bpe_tokens = [
    t for t in most_common
    if (not t[0].startswith("<") and not t[0].endswith(">")) and "->" not in t[0]
][:10]

ast_crosses = [
    cross_dist[token]
    for token, _ in ast_tokens
]
bpe_crosses = [
    cross_dist[token]
    for token, _ in bpe_tokens
]

visualize_perplexities(
    ast_crosses,
    ast_tokens,
    "Cross Entropy of Most Common AST Tokens",
    root_dir / "ast_tokens.png",
)
visualize_perplexities(
    bpe_crosses,
    bpe_tokens,
    "Cross Entropy of Most Common BPE Tokens",
    root_dir / "bpe_tokens.png",
)

mean_crosses = Counter()
for token, losses in cross_dist.items():
    mean_crosses[token] = np.mean(losses)

total_mean = np.mean([np.mean(losses) for losses in cross_dist.values()])

worst_perform = mean_crosses.most_common(10)
for i in range(len(worst_perform)):
    worst_perform[i] = (worst_perform[i][0] + ", " + str(token_cnt[worst_perform[i][0]]), worst_perform[i][1])

best_perform = mean_crosses.most_common()[::-1][:10]
for i in range(len(best_perform)):
    best_perform[i] = (best_perform[i][0] + ", " + str(token_cnt[best_perform[i][0]]), best_perform[i][1])

print(worst_perform)
print(best_perform)
visualize_bar(worst_perform, "Worst Performers", root_dir / "worst_performers.png", total_mean)
visualize_bar(best_perform, "Best Performers", root_dir / "best_performers.png")

# create word clouds
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenized_dataset = load_from_disk(args.tokenized_dataset_dir)# .select(range(1_000))

# get all tokens for internal and external method invocations
internal_tokens = []
external_tokens = []
for example in tqdm(tokenized_dataset, desc="Getting tokens"):
    for i in range(len(example["input_ids"])):
        if example["merged_ast"][i].startswith("<call"):
            token = tokenizer.decode(example["input_ids"][i]).strip()
            if "endoftext" in token:
                continue
            if "internal" in example["merged_ast"][i]:
                internal_tokens.append(token)
            else:
                external_tokens.append(token)

# create word clouds
visualize_word_cloud(internal_tokens, root_dir / "internal_word_cloud.png")
visualize_word_cloud(external_tokens, root_dir / "external_word_cloud.png")