# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['loss_func', 'get_counts', 'perplexed']

# %% ../nbs/00_core.ipynb 2
import datasets
import torch
import transformers

from collections import Counter, defaultdict
from rich.progress import track
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import default_data_collator

# %% ../nbs/00_core.ipynb 4
def loss_func(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # reshape to (batch_size, sequence_length)
    loss = loss.view(*shift_labels.size())
    return loss

# %% ../nbs/00_core.ipynb 5
def get_counts(model, tokenizer, batch, semantic_column: str, return_distributions: bool):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    # input_ids = torch.tensor(batch["input_ids"])
    # attention_mask = torch.tensor(batch["attention_mask"])
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
    loss = loss_func(outputs.logits, input_ids)
    # print(loss.shape)
    # print(input_ids.shape)
    # Add the losses to the counter for each 
    # token in the input
    loss_cnt = defaultdict(list) if return_distributions else Counter()
    token_cnt = Counter()
    for i, ids in enumerate(input_ids):
        for j, token in enumerate(ids[1:]):
            token = tokenizer.decode(token)
            loss_cnt[token] += [loss[i][j].item()] if return_distributions else loss[i][j].item()
            token_cnt[token] += 1

            if semantic_column != None:
                semantic = batch[semantic_column][i]
                loss_cnt[semantic] += [loss[i].item()] if return_distributions else loss[i].item()
                token_cnt[semantic] += 1
    # for i, token in enumerate(input_ids[1:]):
    #     token = tokenizer.decode(token)
    #     loss_cnt[token] += [loss[i].item()] if return_distributions else loss[i].item()
    #     token_cnt[token] += 1
    
    #     if semantic_column != None:
    #         semantic = batch[semantic_column][i]
    #         loss_cnt[semantic] += [loss[i].item()] if return_distributions else loss[i].item()
    #         token_cnt[semantic] += 1
    return loss_cnt, token_cnt

# %% ../nbs/00_core.ipynb 6
def perplexed(
    model: transformers.PreTrainedModel, # The model to calculate the perplexity of.
    dataset: datasets.Dataset, # The dataset to calculate the perplexity on.
    tokenizer: transformers.PreTrainedTokenizer = None, # The tokenizer to use to tokenize the dataset. If not provided, the tokenizer associated with the model will be used.
    column: str = "text", # The column of the dataset to calculate the perplexity on.
    semantic_column: str = None, # The column of the dataset to calculate the semantic perplexity on such as NER tags.
    n_gram: int = 1, # The n-gram to calculate the perplexity on.
    batch_size: int = 1, # The batch size to use when calculating the perplexity.
    device: str = "cuda", # The device to use when calculating the perplexity.
    collate_fn = default_data_collator, # The collate function to use when calculating the perplexity.
    return_tokens: bool = False, # Whether to return the tokens counts along with the perplexity.
    return_distributions: bool = False, # Whether to return the perplexity distributions instead of the perplexity.
): # The perplexity of the model on the dataset or a tuple of the perplexity and the token counts.
    """
    Calculate the perplexity of a model on a dataset.
    """
    if tokenizer is None:
        tokenizer = model.config.tokenizer_class.from_pretrained(model.config.pretrained_model_name_or_path)

    # Tokenize the dataset
    batched = batch_size > 1
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x[column], truncation=True, padding="max_length"),
        batched=batched,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    # Create a dataloader for the dataset
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Calculate the perplexity of the model on the dataset
    total_loss_cnt = defaultdict(list) if return_distributions else Counter()
    total_token_cnt = Counter()
    for batch in track(dataloader, description="Calculating perplexity"):
        # Move the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        loss_cnt, token_cnt = get_counts(model, tokenizer, batch, semantic_column, return_distributions)
        for token, loss in loss_cnt.items():
            total_loss_cnt[token] += loss
        total_token_cnt += token_cnt
    
    # Calculate the perplexity
    perplexity = defaultdict(list) if return_distributions else Counter()
    for token, loss in total_loss_cnt.items():
        if return_distributions:
            perplexity[token] = list(map(lambda x: 2 ** x, loss))
        else:
            perplexity[token] = torch.exp(torch.tensor(loss / total_token_cnt[token])).item()
    
    if return_tokens:
        return perplexity, total_token_cnt
    
    return perplexity
