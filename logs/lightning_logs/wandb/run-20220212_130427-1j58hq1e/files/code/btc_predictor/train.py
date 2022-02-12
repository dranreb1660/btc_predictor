# from gc import callbacks

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import time

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    wandb.login()
except:
    print("wandb not installed")

from btc_predictor.data.prep_and_build_features import Features
import btc_predictor.data.base_dataset as ds

from btc_predictor.data.lit_data_model import BTCPriceDataModule
from btc_predictor.models.lit_model import BTCPricePredictor, BTCRegressorLogger
start = time.time()
base_model_path = '.'

N_EPOCHS = 15
BATCH_SIZE = 4096
seq_length = 200
n_hidden = 128

data = Features()  # instantiate data
train_data, val_data = data.get_train_val_scaled()

train_sequences = ds.create_sequences(
    train_data, target_col='close', seq_length=seq_length)
val_sequences = ds.create_sequences(
    val_data, target_col='close', seq_length=seq_length, t_v='Val')
n_features = train_data.shape[1]


def train():

    data_module = BTCPriceDataModule(train_sequences=train_sequences,
                                     val_sequences=val_sequences, batch_size=BATCH_SIZE)
    data_module.setup()
    samples = next(iter(data_module.val_dataloader()))

    model = BTCPricePredictor(n_features=n_features, n_hidden=n_hidden,
                              batch_size=BATCH_SIZE, seq_length=seq_length, lr=0.0001)
    tb_logger = TensorBoardLogger(
        base_model_path+"/logs/lightning_logs", name='btc-price')
    wanb_logger = WandbLogger(
        'btc1', save_dir=base_model_path+"/logs/lightning_logs", project='btc-multi', )
    logger = wanb_logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=base_model_path+'/logs/checkpoint',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    progress_bar = TQDMProgressBar(refresh_rate=30)
    early_stopping_calback = EarlyStopping(monitor="val_loss", patience=5)
    callbacks = [BTCRegressorLogger(samples),
                 early_stopping_calback, checkpoint_callback, progress_bar]

    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         callbacks=callbacks,
                         max_epochs=N_EPOCHS,
                         auto_lr_find=0.0001,
                         precision=16
                         )

    trainer.tune(model, data_module)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


print("------------------done------------")


if __name__ == "__main__":
    train()
