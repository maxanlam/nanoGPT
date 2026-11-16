## Setup

### 1. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## Raspberry Pi 3 Setup

**Tested setup:** Raspberry Pi 3 Model B, 1 GB RAM, Debian 13 (64-bit), Python 3.13.

### 1. Update the system

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git
```

### 2. Navigate to the project

```bash
cd ~/nanogpt
```

### 3. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch (CPU version)

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0+cpu
```

### 5. Install dependencies

```bash
python -m pip install numpy tiktoken transformers==4.57.6 tqdm
```

### 6. Run Botmind

```bash
python chat.py
```

Generate text:

```bash
python sample_botmind.py --out_dir=out_botmind \
    --start="calm=00 joy=00 sad=00 fear=00 anger=01 lonely=00 curious=01"
```

### 7. Copy updated files from Mac to Pi

Run in the Mac Terminal:

```bash
scp -r ~/nanogpt/* max@raspberrypi.local:~/nanogpt/
```

---

## Shakespeare Character-Level Training

Prepare the dataset (reads all `.txt` files in `data/shakespeare_char/`):

```bash
python3 data/shakespeare_char/prepare.py
```

Start training from scratch:

```bash
python3 train.py config/train_shakespeare_char.py
```

Generate text:

```bash
python3 sample_botmind.py --out_dir=out_botmind \
    --start="calm=00 joy=00 sad=00 fear=00 anger=01 lonely=00 curious=01"
```

### Resume Training

Set `init_from = 'resume'` in `config/train_shakespeare_char.py`, then run the training command again.

**Raspberry Pi note:** Training on a Pi 3 with 1 GB RAM is very resource-intensive. Train on a more powerful computer and transfer the model files to the Pi for inference.