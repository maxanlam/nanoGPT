
"""
Prepare the botmind dataset for character-level language modeling.
Special tokens:
    <startoftext|>
    <endoftext|>
All other characters are encoded individually.
"""

import os
import pickle
import re
import numpy as np

here = os.path.dirname(__file__)

START = "<startoftext|>"
END = "<endoftext|>"

with open(os.path.join(here, "input.txt"), "r", encoding="utf-8") as f:
    data = f.read()

print(f"Dataset length: {len(data):,} characters")

# Split into complete examples, assuming examples are separated
# by one or more blank lines.
examples = [x.strip() for x in re.split(r"\n\s*\n", data) if x.strip()]

if len(examples) < 2:
    raise ValueError("Need at least two examples separated by blank lines.")

# 90/10 split by complete examples, not arbitrary characters.
split = max(1, int(len(examples) * 0.9))
train_data = "\n\n".join(examples[:split])
val_data = "\n\n".join(examples[split:])

# Remove special markers before building the character vocabulary.
clean_data = data.replace(START, "").replace(END, "")
chars = sorted(set(clean_data))

# Character IDs.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Reserve two additional IDs for special tokens.
START_ID = len(chars)
END_ID = len(chars) + 1

vocab_size = len(chars) + 2

# Recognize special markers as individual tokens.
pattern = re.compile(
    f"({re.escape(START)}|{re.escape(END)})"
)

def encode(text):
    ids = []
    parts = pattern.split(text)

    for part in parts:
        if not part:
            continue

        if part == START:
            ids.append(START_ID)
        elif part == END:
            ids.append(END_ID)
        else:
            ids.extend(stoi[ch] for ch in part)

    return ids


def decode(ids):
    result = []

    for idx in ids:
        if idx == START_ID:
            result.append(START)
        elif idx == END_ID:
            result.append(END)
        else:
            result.append(itos[int(idx)])

    return "".join(result)


# Encode train and validation sets.
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

print(f"Vocabulary size: {vocab_size}")
print(f"Training tokens: {len(train_ids):,}")
print(f"Validation tokens: {len(val_ids):,}")
print(f"START token ID: {START_ID}")
print(f"END token ID: {END_ID}")

# Save binary datasets.
train_ids.tofile(os.path.join(here, "train.bin"))
val_ids.tofile(os.path.join(here, "val.bin"))

# Save metadata.
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
    "start_id": START_ID,
    "end_id": END_ID,
}

with open(os.path.join(here, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Dataset preparation complete.")
