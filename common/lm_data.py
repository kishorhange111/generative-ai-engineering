"""
Data helpers for supervised fine-tuning of causal (decoder-only) language models.

Two ideas every SFT script in this repo relies on:

1. Loss masking. A training sequence is  prompt + answer.  We want the model to
   learn to *write the answer*, not to reproduce the prompt, so every prompt
   position gets the label -100. PyTorch's cross-entropy ignores -100, so
   those tokens give context but no gradient.

2. Dynamic padding. Sequences in a batch have different lengths; we pad each
   batch only to its own longest sequence (not to a global maximum), which
   saves a lot of wasted compute on short examples.

The helpers return plain Python lists so they can be tested without PyTorch;
the training scripts wrap the result in `torch.tensor`.
"""

# Label value ignored by torch.nn.CrossEntropyLoss (its default ignore_index).
IGNORE_INDEX = -100


def masked_labels(prompt_ids, full_ids):
    """Labels for `full_ids` that only train on the tokens after the prompt.

    prompt_ids : token ids of the prompt alone (ends with the assistant header).
    full_ids   : token ids of prompt + answer (possibly truncated to a max length).

    The result always has exactly `len(full_ids)` entries, even when truncation
    cut into the answer (or the prompt).

    >>> masked_labels([1, 2], [1, 2, 3, 4])
    [-100, -100, 3, 4]
    """
    return ([IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):]))[:len(full_ids)]


def pad_batch(batch, pad_token_id):
    """Right-pad a list of {"input_ids", "labels"} dicts to the longest sequence.

    Returns a dict of lists of lists with three keys:
        input_ids      - padded with `pad_token_id`
        labels         - padded with -100 (padding never contributes to the loss)
        attention_mask - 1 for real tokens, 0 for padding (the model skips them)
    """
    width = max(len(b["input_ids"]) for b in batch)

    def pad(seq, value):
        return list(seq) + [value] * (width - len(seq))

    return {
        "input_ids": [pad(b["input_ids"], pad_token_id) for b in batch],
        "labels": [pad(b["labels"], IGNORE_INDEX) for b in batch],
        "attention_mask": [pad([1] * len(b["input_ids"]), 0) for b in batch],
    }
