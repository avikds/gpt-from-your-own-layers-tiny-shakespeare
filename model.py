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

# Step 2 - get_batch
def get_batch(data, block_size, batch_size, generator=None):
    """Random (batch_size, block_size) chunks x of data and their next-token targets y."""

    # Generate random starting positions for each chunk.
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
        generator=generator
    )

    # Extract input sequences of length block_size.
    x = torch.stack([
        data[int(i):int(i) + block_size] for i in ix
    ])

    # Extract target sequences shifted one character ahead.
    y = torch.stack([
        data[int(i) + 1:int(i) + block_size + 1] for i in ix
    ])

    return x, y

# Step 3 - BigramLanguageModel
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """One embedding table: row i holds the logits for the token after token i."""
        super().__init__()

        # Each token ID maps directly to logits over the entire vocabulary.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits of shape (B, T, vocab_size), loss or None)."""

        # Look up the logits corresponding to each input token.
        # Input shape: (B, T)
        # Output shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None

        # Calculate cross-entropy loss when target tokens are provided.
        if targets is not None:
            B, T, C = logits.shape

            # Flatten logits and targets for cross-entropy calculation.
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

# Step 4 - generate
def generate(model, idx, max_new_tokens, block_size, generator=None):
    """Append max_new_tokens sampled tokens to idx (B, T), cropping the context to block_size."""

    # Disable gradient calculation during text generation.
    with torch.no_grad():

        # Generate one token at a time.
        for _ in range(max_new_tokens):

            # Keep only the most recent block_size tokens as context.
            idx_cond = idx[:, -block_size:]

            # Get predictions from the model.
            logits, _ = model(idx_cond)

            # Select the logits for the last time step.
            logits = logits[:, -1, :]

            # Convert logits into a probability distribution.
            probs = F.softmax(logits, dim=-1)

            # Sample one next token for each sequence in the batch.
            idx_next = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # Append the sampled token to the existing sequence.
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Step 5 - train
def train(model, data, block_size, batch_size, steps, lr=1e-3, generator=None):
    """AdamW training on random batches; returns each step's loss as a float."""

    # Initialize the AdamW optimizer once before training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Store the training loss from each optimization step.
    losses = []

    # Perform the specified number of training steps.
    for step in range(steps):

        # Sample a random batch of input sequences and their targets.
        xb, yb = get_batch(
            data,
            block_size,
            batch_size,
            generator=generator
        )

        # Forward pass: calculate predictions and cross-entropy loss.
        logits, loss = model(xb, yb)

        # Clear previously accumulated gradients.
        optimizer.zero_grad(set_to_none=True)

        # Backpropagate the loss to calculate gradients.
        loss.backward()

        # Update model parameters using AdamW.
        optimizer.step()

        # Record the current loss as a Python float.
        losses.append(float(loss.item()))

    return losses

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

