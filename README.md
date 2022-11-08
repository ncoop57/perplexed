perplexed
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This library is based on the idea from Andrej Karpathy on understanding
the failure cases of a model by looking at the worst predictions.
Specifically, this library focuses on calculating the perplexity of
Large Language Models (LLMs) such as GPT-2 and BERT. The idea is to
calculate the perplexity of a model on a dataset at the per token level.
This allows us to understand where the model is perplexed and where it
is not. This is useful for debugging and understanding the model.

## Install

``` sh
pip install perplexed
```

## How to use

### Using the API

[`perplexed`](https://ncoop57.github.io/perplexed/core.html#perplexed)
is designed to work with the HuggingFace ecosystem and is built on top
of the `transformers` and `datasets` libraries. The API is designed to
be simple and easy to use. The main function is
[`perplexed`](https://ncoop57.github.io/perplexed/core.html#perplexed)
which takes in a model, dataset, and tokenizer and returns a simple
Counter object with the perplexity of each token in the dataset. Here is
an example of how to use it:

``` python
from perplexed.core import perplexed

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").select(range(100))
# filter out empty strings
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

perplexity_cnt = perplexed(model, dataset, tokenizer=tokenizer, column="text", batch_size=1, device="cpu")
perplexity_cnt.most_common(10)
```

    Found cached dataset wikitext (/home/nathan/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126)
    Loading cached processed dataset at /home/nathan/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/cache-68eb731029328d8b.arrow
    Loading cached processed dataset at /home/nathan/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/cache-1c1cd85efcee4db8.arrow

    [(' wired', 60983688.0),
     (' 768', 21569838.0),
     (' shatter', 12281687.0),
     (' unsett', 8289435.0),
     (' ignited', 6605209.0),
     (' Tanz', 4834899.0),
     (' Influence', 4153321.75),
     (' Career', 4064189.0),
     (' Television', 2325870.75),
     (' Moral', 2243574.5)]
