# Language modelling

This repo contains code related to language modelling.

### Character-level language models and datasets

Table of language models and their associated dataset:

| **Language Model**                      | **Dataset**                                           |
| --------------------------------------- | ----------------------------------------------------- |
| [Bigram](lm/models/bigram.py)           | [CharDataset](lm/datasets/char_dataset.py)            |
| [MLP](lm/models/mlp.py)                 | [MultiCharDataset](lm/datasets/multi_char_dataset.py) |
| [RNN](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [GRU](lm/models/rnn.py)                 | [SequenceDataset](lm/datasets/sequence_dataset.py)    |
| [Transformer](lm/models/transformer.py) | [SequenceDataset](lm/datasets/sequence_dataset.py)    |

### Tokenizers

- [BasicTokenizer](lm/tokenization/basic_tokenizer.py)
- [RegexTokenizer](lm/tokenization/regex_tokenizer.py)

### Normalization layers

- [BatchNorm](lm/normalization/batch_norm.py)
- [LayerNorm](lm/normalization/layer_norm.py)

### Manual backpropagation

- [Manual backward pass](lm/backprop/backward_test.py)

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

## References

- [makemore](https://github.com/karpathy/makemore),
  [nanoGPT](https://github.com/karpathy/nanoGPT), and
  [minbpe](https://github.com/karpathy/minbpe) by Andrej Karpathy. These are awesome and
  I highly recommend checking them out!
