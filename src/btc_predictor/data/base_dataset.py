from Btc_Predictor.btc_predictor.data.prep_n_build_features import Features


# Creat sequences, take n sequences and predict the n+1st term
def create_sequences(data: pd.DataFrame, target_col, seq_length):
    sequences = []
    data_sz = len(data)

    for i in tqdm(range(data_sz - seq_length)):
        sequence = data[i:i+seq_length]
        label_pos = i+seq_length
        label = data.iloc[label_pos][target_col]

        sequences.append((sequence, label))

    return sequences


seq_length = 200
data = Features()  # instantiate data
tain_data, val_data = data.get_train_val_scaled()

train_sequences = create_sequences(
    train_data, target_col='close', seq_length=seq_length)
val_sequences = create_sequences(
    val_data, target_col='close', seq_length=seq_length)


class BTCDataset(Dataset):
    def __init__(self, sequences) -> None:
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=tensor(sequence),
            label=tensor(label).float()
        )
