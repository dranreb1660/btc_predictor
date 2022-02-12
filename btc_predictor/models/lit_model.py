from turtle import pd
import pytorch_lightning as pl
import torch
from torch import nn
import pandas as pd
import wandb

from btc_predictor.models.base_model import PricePredictionModel

# creating a lightning module

# wandb.init()


class BTCPricePredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_hidden: int, batch_size, seq_length, lr) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.model = PricePredictionModel(self.n_features, self.n_hidden)
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
                 logger=True, on_step=False, on_epoch=True)
        return outputs

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        self.log('test_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        dummyImput = torch.zeros(
            (1, self.seq_length, self.n_features), device=self.device)
        model_filename = f'model_{str(self.current_epoch)}.onnx'
        # torch.onnx.export(self, dummyImput, model_filename)
        # wandb.save(model_filename)

        flattened_outputs = torch.flatten(
            torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {'valid/logits': wandb.Histogram(flattened_outputs.to('cpu')),
             'epoch': self.current_epoch}
        )

    def test_epoch_end(self, test_step_outputs):
        dummyImput = torch.zeros(
            (self.batch_size, self.seq_length, self.n_features), device=self.device)
        model_filename = f'model_final.onnx'
        # torch.onnx.export(self, dummyImput, model_filename)
        # wandb.save(model_filename)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class BTCRegressorLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32) -> None:
        super().__init__()
        self.val_seqs = val_samples['sequence']
        self.val_labels = val_samples['label']
        self.val_seqs = self.val_seqs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_seqs = self.val_seqs.to(device=pl_module.device)

        logits = pl_module(val_seqs)

        df = pd.DataFrame(self.val_seqs.squeeze())
        print([logits[1].item()])

        df['preds'] = [logits[1].item()]*len(df)
        df['truth'] = [self.val_labels]*len(df)
        data = df.values

        trainer.logger.experiment.log({
            'examples': wandb.Table(data=data,
                                    columns=['day_of_week', 'day_of_month', 'week_of_year', 'month', 'open', 'high', 'low', 'close_change', 'close', 'preds', 'price'])
        })
