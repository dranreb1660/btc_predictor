import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os

from btc_predictor.data.base_dataset import BTCDataset

# wraping the above into a pl datamodule


class BTCPriceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, batch_sz=8):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.batch_size = batch_sz

    def setup(self):
        self.train_dataset = BTCDataset(self.train_sequences)
        self.val_dataset = BTCDataset(self.val_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, bs=self.batch_size,
            shuffle=False, num_workers=os.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, bs=1,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, bs=1,
            num_workers=1
        )
