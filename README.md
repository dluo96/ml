# Character-level Language Models

This repo contains code for training and sampling from character-level language models.
It is based on [makemore](https://github.com/karpathy/makemore) and
[nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, which I highly
recommend!

Table of language models and their associated dataset:

| **Language Model**                      | **Dataset**                                           |
| --------------------------------------- | ----------------------------------------------------- |
| [Bigram](lm/models/bigram.py)           | [CharDataset](lm/datasets/char_dataset.py)            |
| [MLP](lm/models/mlp.py)                 | [MultiCharDataset](lm/datasets/multi_char_dataset.py) |
| [RNN](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [GRU](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [Transformer](lm/models/transformer.py) | [SequenceDataset](lm/datasets/sequence_dataset.py)    |

## Getting started

### Installation

- [Install PyTorch](https://pytorch.org/get-started/locally/) (the only requirement).
- Install the package: `pip install -e .`

### Training

- Run the training script: `python main.py`

### Tests

Run all tests:

```bash
python -m unittest discover --verbose --failfast --start-directory lm/ --pattern '*_test.py'
```
