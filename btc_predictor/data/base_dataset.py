from numpy import dtype
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from btc_predictor.data.prep_and_build_features import Features


# Creat sequences, take n sequences and predict the n+1st term
def create_sequences(data: pd.DataFrame, target_col, seq_length, t_v='Train'):
    print(f'\n---------Creating {t_v} sequences of {seq_length}--------\n')
    sequences = []
    data_size = len(data)
    print(target_col)
    for i in tqdm(range(data_size - seq_length)):
        sequence = data[i:i+seq_length]
        label_pos = i+seq_length
        label = data.iloc[label_pos][target_col]

        sequences.append((sequence, label))
    print(
        f'\n------------- {len(sequences)} Sequences creation compleated---------\n')
    return sequences


class BTCDataset(Dataset):
    def __init__(self, sequences) -> None:
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=torch.tensor(sequence.to_numpy(), dtype=torch.float),
            label=torch.tensor(label).float()
        )
