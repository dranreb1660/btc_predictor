# from gc import callbacks
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from btc_predictor.data.prep_and_build_features import Features
import btc_predictor.data.base_dataset as ds

from btc_predictor.data.lit_data_model import BTCPriceDataModule
from btc_predictor.models.lit_model import BTCPricePredictor

base_model_path = '.'

N_EPOCHS = 12
BATCH_SIZE = 4096
seq_length = 200

data = Features()  # instantiate data
train_data, val_data = data.get_train_val_scaled()

train_sequences = ds.create_sequences(
    train_data, target_col='close', seq_length=seq_length)
val_sequences = ds.create_sequences(
    val_data, target_col='close', seq_length=seq_length)
n_features = train_data.shape[1]


def main():

    data_module = BTCPriceDataModule(train_sequences=train_sequences,
                                     val_sequences=val_sequences, batch_sz=BATCH_SIZE)

    model = BTCPricePredictor(n_features=n_features, lr=0.0001)
    logger = TensorBoardLogger(
        base_model_path+"/logs/lightning_logs", name='btc-price')

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
    callbacks = [ModelSummary(max_depth=-1),
                 early_stopping_calback, checkpoint_callback, progress_bar]

    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         callbacks=callbacks,
                         max_epochs=N_EPOCHS,
                         progress_bar_refresh_rate=30,
                         auto_lr_find=0.0001,
                         precision=16
                         )

    trainer.tune(model, data_module)
    trainer.fit(model, data_module)
    trainer.test(model, data)


if __name__ == "__main__":
    main()
