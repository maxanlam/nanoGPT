    ## Setup

    Create and activate the virtual environment:

        python3 -m venv .venv
        source .venv/bin/activate

    Install dependencies:

        python3 -m pip install -r requirements.txt

    ## Shakespeare Character-Level Training

    Prepare the dataset (reads all `.txt` files in `data/shakespeare_char/`):

        python3 data/shakespeare_char/prepare.py

    Start training from scratch:

        python3 train.py config/train_shakespeare_char.py

    Generate text:

        python3 sample.py --out_dir=out-shakespeare-char

    ## or Resume Training:

    Set `init_from = 'resume'` in `config/train_shakespeare_char.py`, then run the training command again.

