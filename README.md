# Language modelling

This repo contains code related to language modelling.

### Character-level generative language models and datasets

| **Language Model**                                                     | **Dataset**                                                      |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [Bigram](lm/generative/models/bigram.py)                               | [CharDataset](lm/generative/datasets/char_dataset.py)            |
| [MLP](lm/generative/models/mlp.py)                                     | [MultiCharDataset](lm/generative/datasets/multi_char_dataset.py) |
| [RNN](lm/generative/models/rnn.py)                                     | [SequenceDataset](lm/generative/datasets/sequence_dataset.py)    |
| [GRU](lm/generative/models/rnn.py)                                     | [SequenceDataset](lm/generative/datasets/sequence_dataset.py)    |
| [Transformer Decoder (GPT-style)](lm/generative/models/transformer.py) | [SequenceDataset](lm/generative/datasets/sequence_dataset.py)    |

### Language representation models and datasets

| **Language Model**                                         | **Dataset**                          |
| ---------------------------------------------------------- | ------------------------------------ |
| [Transformer Encoder (BERT-style)](lm/bert/transformer.py) | [MLMDataset](lm/bert/mlm_dataset.py) |

### Tokenizers

- [BasicTokenizer](lm/tokenization/basic_tokenizer.py)
- [RegexTokenizer](lm/tokenization/regex_tokenizer.py)

### Positional Encodings

- [Rotary Position Embedding (RoPE)](lm/pos_embd/rope.py)

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
- [A Programmer's Intro To Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
- [Integer Tokenization is Insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/)
- [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- [Pytorchic BERT](https://github.com/dhlee347/pytorchic-bert)
- [simple-bert](https://github.com/lukemelas/simple-bert)
