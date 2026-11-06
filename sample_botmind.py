
"""
sample_botmind.py

Generate emotional text using the Botmind nanoGPT model.

Example:
python sample_botmind.py --out_dir=out_botmind \
    --start="calm=00 joy=00 sad=00 fear=00 anger=01 lonely=00 curious=01"

The tokenizer must match the one used during training.
"""

import os
import pickle
import re
import torch
from contextlib import nullcontext

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

out_dir = "out_botmind"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = "float32"

temperature = 0.3
max_new_tokens = 256

START = "<startoftext|>"
END = "<endoftext|>"

# ------------------------------------------------------------
# Load tokenizer metadata
# ------------------------------------------------------------

meta_path = os.path.join("data", "botmind", "meta.pkl")

with open(meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]

# Adapt these names if your prepare.py uses different keys.
start_id = meta["start_id"]
end_id = meta["end_id"]

vocab_size = len(stoi) + 2

pattern = re.compile(
    f"({re.escape(START)}|{re.escape(END)})"
)

def encode(text):
    ids = []

    for part in pattern.split(text):
        if not part:
            continue
        elif part == START:
            ids.append(start_id)
        elif part == END:
            ids.append(end_id)
        else:
            ids.extend(stoi[c] for c in part)

    return ids


def decode(ids):
    result = []

    for idx in ids:
        idx = int(idx)

        if idx == start_id:
            result.append(START)
        elif idx == end_id:
            result.append(END)
        else:
            result.append(itos[idx])

    return "".join(result)


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

# Import the model architecture from nanoGPT.
from model import GPTConfig, GPT

checkpoint_path = os.path.join(out_dir, "ckpt.pt")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model_args = checkpoint["model_args"]
gptconf = GPTConfig(**model_args)

model = GPT(gptconf)

state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."

for key, value in list(state_dict.items()):
    if key.startswith(unwanted_prefix):
        state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# ------------------------------------------------------------
# Input emotional values
# ------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start",
    type=str,
    default="calm=00 joy=00 sad=00 fear=00 anger=00 lonely=00 curious=00"
)

args = parser.parse_args()

prompt = args.start.strip()

# Add START token if the user didn't provide it.
if not prompt.endswith(START):
    prompt += " " + START

prompt_ids = encode(prompt)

x = torch.tensor(
    prompt_ids,
    dtype=torch.long,
    device=device
)[None, :]

# ------------------------------------------------------------
# Generate text
# ------------------------------------------------------------

print("\nBotmind is thinking...\n")

with torch.no_grad():
    for _ in range(max_new_tokens):

        x_cond = x[:, -model_args["block_size"]:]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        next_id = next_token.item()

        if next_id == end_id:
            break

        x = torch.cat((x, next_token), dim=1)

generated = decode(x[0].tolist())

# Only show the generated text, not the input prompt.
if START in generated:
    generated = generated.split(START, 1)[1]

if END in generated:
    generated = generated.split(END, 1)[0]

print("Botmind:")
print(generated.strip())
print()
