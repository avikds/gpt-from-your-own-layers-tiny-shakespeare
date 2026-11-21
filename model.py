"""
GPT From Your Own Layers: Tiny Shakespeare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import os
import tempfile
import urllib.request
import torch


def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare."""

    # URL and local path for the Tiny Shakespeare dataset.
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")

    # Download the dataset only if it is not already available locally.
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        urllib.request.urlretrieve(url, file_path)

    # Read the complete dataset using UTF-8 encoding.
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create a sorted vocabulary of all unique characters.
    chars = sorted(list(set(text)))

    # Build character-to-integer and integer-to-character mappings.
    stoi = {ch: i for i, ch in enumerate(chars)}

    # Encode a string into a list of integer IDs.
    encode = lambda s: [stoi[c] for c in s]

    # Decode a list of integer IDs back into a string.
    decode = lambda ids: "".join(chars[int(i)] for i in ids)

    # Encode the entire text into a one-dimensional torch.long tensor.
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split the encoded data into training and validation sets.
    split_idx = int(train_frac * len(data))

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return chars, encode, decode, train_data, val_data

# Step 2 - get_batch (not yet solved)
# TODO: implement

# Step 3 - BigramLanguageModel (not yet solved)
# TODO: implement

# Step 4 - generate (not yet solved)
# TODO: implement

# Step 5 - train (not yet solved)
# TODO: implement

# Step 6 - estimate_loss (not yet solved)
# TODO: implement

# Step 7 - average_past (not yet solved)
# TODO: implement

# Step 8 - Head (not yet solved)
# TODO: implement

# Step 9 - MultiHeadAttention (not yet solved)
# TODO: implement

# Step 10 - FeedForward (not yet solved)
# TODO: implement

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

