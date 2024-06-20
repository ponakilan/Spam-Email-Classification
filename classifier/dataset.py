# Built-in dependencies
from typing import Tuple

# External dependencies
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


class TextDataset(Dataset):
    def __init__(self, train_csv: str, valid_csv: str, text_ind: int, label_ind: int, max_len: int, train: bool):
        if train:
            self.data = pd.read_csv(train_csv)
        else:
            self.data = pd.read_csv(valid_csv)
        self.text_ind = text_ind
        self.label_ind = label_ind
        self.max_len = max_len

        self.tokens_to_idx = {}
        self.tokenizer = get_tokenizer("basic_english")

        train_df = pd.read_csv(train_csv)
        for text in train_df.iloc[:, self.text_ind]:
            self.convert_to_vector(text, max_len)

        valid_df = pd.read_csv(valid_csv)
        for text in valid_df.iloc[:, self.text_ind]:
            self.convert_to_vector(text, max_len)

    def __len__(self):
        return self.data.shape[0]

    def convert_to_vector(self, text: str, max_len: int) -> torch.Tensor:
        tokens = self.tokenizer(text)

        indices = []
        for token in tokens:
            keys = self.tokens_to_idx.keys()
            if token not in keys:
                self.tokens_to_idx[token] = len(keys) + 1
            indices.append(self.tokens_to_idx[token])

        len_tokens = len(tokens)
        if len_tokens < max_len:
            indices.extend([0 for i in range(max_len - len_tokens)])
        elif len_tokens > max_len:
            indices = indices[:max_len]

        return torch.tensor(indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.data.iloc[index, self.text_ind]
        label = self.data.iloc[index, self.label_ind]

        X = self.convert_to_vector(text, self.max_len)
        y = torch.tensor(label)

        return X, y
