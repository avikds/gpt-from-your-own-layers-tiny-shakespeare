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

# Step 6 - estimate_loss
def estimate_loss(model, splits, block_size, batch_size, eval_iters, generator=None):
    """Mean loss over eval_iters batches for each named split, in evaluation mode, as floats."""

    # Switch the model to evaluation mode.
    model.eval()

    # Dictionary to store the average loss for each split.
    out = {}

    # Disable gradient tracking during evaluation.
    with torch.no_grad():

        # Evaluate each dataset split in dictionary order.
        for split, data in splits.items():

            total_loss = 0.0

            # Calculate loss over eval_iters random batches.
            for _ in range(eval_iters):

                # Sample a batch of inputs and next-token targets.
                xb, yb = get_batch(
                    data,
                    block_size,
                    batch_size,
                    generator=generator
                )

                # Perform a forward pass and calculate the loss.
                logits, loss = model(xb, yb)

                # Accumulate the batch loss.
                total_loss += loss.item()

            # Store the average loss as a Python float.
            out[split] = total_loss / eval_iters

    # Restore the model to training mode.
    model.train()

    return out

# Step 7 - average_past
def average_past(x):
    """Position t of the output is the mean of x[:, :t + 1], computed with a masked softmax and a matrix multiply."""

    # Get the sequence length from the input tensor.
    B, T, C = x.shape

    # Initialize a (T, T) weight matrix with zeros.
    weights = x.new_zeros((T, T))

    # Create a lower-triangular causal mask.
    # Each position can attend only to itself and previous positions.
    causal_mask = torch.tril(
        torch.ones((T, T), device=x.device, dtype=torch.bool)
    )

    # Set positions above the diagonal to negative infinity.
    # These positions will receive zero probability after softmax.
    weights = weights.masked_fill(~causal_mask, float("-inf"))

    # Apply softmax along the last dimension.
    # All allowed positions receive equal weights.
    weights = F.softmax(weights, dim=-1)

    # Compute the weighted average using matrix multiplication.
    # Output shape: (B, T, C)
    out = weights @ x

    return out

# Step 8 - Head
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        """Key, query and value projections, a registered tril buffer, and dropout on the attention weights."""
        super().__init__()

        # Linear projections for keys, queries, and values.
        # Bias is disabled as specified.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Register a lower-triangular causal mask.
        # This ensures that each token can only attend to itself
        # and the tokens appearing before it.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        # Dropout applied to the attention weights.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, T, n_embd) -> (B, T, head_size): scaled, causally masked attention."""

        # Get batch size, sequence length, and embedding dimensions.
        B, T, C = x.shape

        # Generate query, key, and value representations.
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Calculate scaled dot-product attention scores.
        # Shape: (B, T, T)
        scores = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)

        # Apply the causal mask to prevent attending to future positions.
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert attention scores into probability distributions.
        weights = F.softmax(scores, dim=-1)

        # Apply dropout to the attention weights.
        weights = self.dropout(weights)

        # Compute the weighted sum of the value vectors.
        # Output shape: (B, T, head_size)
        out = weights @ v

        return out

# Step 9 - MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        """num_heads Heads in an nn.ModuleList, a projection back to n_embd, and dropout."""
        super().__init__()

        # Create multiple independent self-attention heads.
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        # Project the concatenated head outputs back to n_embd dimensions.
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        # Apply dropout after the output projection.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Concatenate the heads' outputs, project, dropout."""

        # Run every attention head on the same input.
        # Concatenate their outputs along the embedding dimension.
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        # Project the concatenated output back to the original
        # embedding dimension.
        out = self.proj(out)

        # Apply dropout to the projected output.
        out = self.dropout(out)

        return out

# Step 10 - FeedForward (not yet solved)
# TODO: implement

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

