# Language Modelling

This repo contains code for training and sampling from character-level language models.

Table of language models and their associated dataset:

| **Language Model**                      | **Dataset**                                           |
| --------------------------------------- | ----------------------------------------------------- |
| [Bigram](lm/models/bigram.py)           | [CharDataset](datasets/char_dataset.py)               |
| [MLP](lm/models/mlp.py)                 | [MultiCharDataset](lm/datasets/multi_char_dataset.py) |
| [RNN](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [GRU](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [Transformer](lm/models/transformer.py) | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
