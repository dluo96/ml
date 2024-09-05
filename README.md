# Character-level Language Models

This repo contains code for training and sampling from character-level language models.
It is based on [makemore](https://github.com/karpathy/makemore) by Andrej Karpathy,
which I highly recommend!

Table of language models and their associated dataset:

| **Language Model**                      | **Dataset**                                           |
| --------------------------------------- | ----------------------------------------------------- |
| [Bigram](lm/models/bigram.py)           | [CharDataset](lm/datasets/char_dataset.py)            |
| [MLP](lm/models/mlp.py)                 | [MultiCharDataset](lm/datasets/multi_char_dataset.py) |
| [RNN](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [GRU](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [Transformer](lm/models/transformer.py) | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
