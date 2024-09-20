import random

from datasets import load_dataset
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    def __init__(self, words: list[str]):
        self.words = words
        self.unique_chars = ["<CLS>", "<SEP>", "<MASK>"] + sorted(list(set("".join(words))))  # fmt: skip
        self.ctoi = {c: i for i, c in enumerate(self.unique_chars)}
        self.itoc = {i: c for i, c in enumerate(self.unique_chars)}

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        """Retrieves the input and target tensors for a given index.

        Note that the special tokens have the following token IDs:
            - "<CLS>": 0
            - "<SEP>": 1
            - "<MASK>": 2

        For example, suppose we have the idx representing the word "emma". Then, the
        input might be something like:
            - Input: [0, 7, 15, 2, 3, 1]
            - Target positions: [3]
            - Target labels: [15]

        For the BERT model, this means that:
            - When the to-be-predicted token in ["<CLS>", "e", "m", "m", "a", "<SEP>"]
                is the second "m", which has position 3, the target label is 15.
        """

    def _replace_mlm_tokens(
        self,
        tokens: list[str],
        eligible_positions: list[int],
        num_mlm_preds: int,
    ) -> tuple[list[str], list[tuple[int, str]]]:
        # Create a copy of tokens and replace some of them by <MASK> or random tokens
        mlm_input_tokens = [t for t in tokens]

        # Store positions and labels of to-be-predicted tokens
        pred_positions_and_labels = []

        # Shuffle to avoid always selecting from the first few tokens
        random.shuffle(eligible_positions)

        # Iterate over eligible positions
        for pos in eligible_positions:
            # If we have already chosen `num_mlm_preds` tokens to predict, we are done
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break

            # 80% of the time: replace the token with the <MASK> token.
            # 10% of the time: replace the token with a random token.
            # 10% of the time: keep the token unchanged.
            p = random.random()
            if p < 0.8:
                masked_token = "<MASK>"
            elif p < 0.9:
                masked_token = random.choice(list(self.ctoi.keys()))
            else:
                masked_token = tokens[pos]

            # Replace the token with the masked token
            mlm_input_tokens[pos] = masked_token

            # Store the position and label of the to-be-predicted token
            pred_positions_and_labels.append((pos, tokens[pos]))

        return mlm_input_tokens, pred_positions_and_labels

    def _get_mlm_data_from_tokens(
        self, tokens: list[str]
    ) -> tuple[list[int], list[int], list[int]]:
        # Filter out <CLS> and <SEP> tokens to get positions of tokens that may or may
        # not be used for prediction in MLM
        eligible_positions = []
        for position, token in enumerate(tokens):
            # Special tokens are not predicted in MLM
            if token in ["<CLS>", "<SEP>"]:
                continue
            eligible_positions.append(position)

        # 15% of tokens are randomly chosen for prediction in MLM (BERT's pre-training task #1)
        num_mlm_preds = max(1, round(len(tokens) * 0.15))

        # Get masked input tokens and labels
        mlm_input_tokens, pred_positions_and_labels = self._replace_mlm_tokens(
            tokens,
            eligible_positions,
            num_mlm_preds,
        )

        # Sort by position to undo the shuffling from earlier
        pred_positions_and_labels = sorted(
            pred_positions_and_labels, key=lambda x: x[0]
        )

        # Separate the positions and labels for the to-be-predicted tokens
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

        # Convert tokens to token IDs
        token_ids = [self.ctoi[token] for token in mlm_input_tokens]
        label_token_ids = [self.ctoi[label] for label in mlm_pred_labels]

        return token_ids, pred_positions, label_token_ids
