import pytorch_lightning as pl
import torch
from torch import nn

from btc_predictor.models.base_model import PricePredictionModel

# creating a lightning module


class BTCPricePredictor(pl.LightningModule):
    def __init__(self, n_features: int, lr) -> None:
        super().__init__()
        self.model = PricePredictionModel(n_features)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0

        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log('val_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log('test_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
