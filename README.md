# Language modelling

This repo contains code related to language modelling.

### Character-level generative language models and datasets

| **Language Model**                                                        | **Dataset**                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [Bigram](ml/lm/generative/models/bigram.py)                               | [CharDataset](ml/lm/generative/datasets/char_dataset.py)            |
| [MLP](ml/lm/generative/models/mlp.py)                                     | [MultiCharDataset](ml/lm/generative/datasets/multi_char_dataset.py) |
| [RNN](ml/lm/generative/models/rnn.py)                                     | [SequenceDataset](ml/lm/generative/datasets/sequence_dataset.py)    |
| [GRU](ml/lm/generative/models/rnn.py)                                     | [SequenceDataset](ml/lm/generative/datasets/sequence_dataset.py)    |
| [Transformer Decoder (GPT-style)](ml/lm/generative/models/transformer.py) | [SequenceDataset](ml/lm/generative/datasets/sequence_dataset.py)    |

### Language representation models and datasets

| **Language Model**                                            | **Dataset**                             |
| ------------------------------------------------------------- | --------------------------------------- |
| [Transformer Encoder (BERT-style)](ml/lm/bert/transformer.py) | [MLMDataset](ml/lm/bert/mlm_dataset.py) |

### Tokenizers

- [BasicTokenizer](ml/lm/tokenization/basic_tokenizer.py)
- [RegexTokenizer](ml/lm/tokenization/regex_tokenizer.py)

### Positional Encodings

- [Rotary Position Embedding (RoPE)](ml/lm/pos_embd/rope.py)

### Normalization layers

- [BatchNorm](ml/normalization/batch_norm.py)
- [LayerNorm](ml/normalization/layer_norm.py)

### Manual backpropagation

- [Manual backward pass](ml/backpropagation/backward_test.py)

### Optimizers

- [Adam](ml/optimizers/adam.py)

### Autoencoders

- [Autoencoder](ml/autoencoders/autoencoder.py)
- [Variational Autoencoder (VAE)](ml/autoencoders/variational_autoencoder.py)

## Getting started

### Installation

- [Install PyTorch](https://pytorch.org/get-started/locally/) (the only requirement).
- Install the package: `pip install -e .`

### Training

- Run the training script: `python main.py`

### Tests

Run all tests:

```bash
python -m unittest discover --verbose --failfast --start-directory ml/ --pattern '*_test.py'
```

## References

- [makemore](https://github.com/karpathy/makemore),
  [nanoGPT](https://github.com/karpathy/nanoGPT), and
  [minbpe](https://github.com/karpathy/minbpe) by Andrej Karpathy. These are awesome and
  I highly recommend checking them out!
- [A Programmer's Intro To Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
- [Integer Tokenization is Insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/)
- [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- [Pytorchic BERT](https://github.com/dhlee347/pytorchic-bert)
- [simple-bert](https://github.com/lukemelas/simple-bert)
