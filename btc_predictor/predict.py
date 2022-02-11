import time
from btc_predictor.models.train import *
from btc_predictor.data.base_dataset import BTCDataset
from btc_predictor.models.lit_model import BTCPricePredictor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from tqdm import tqdm


scaler = data.scaler
de_scaler = MinMaxScaler()


def descale(de_scaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return de_scaler.inverse_transform(values_2d).flatten()


trained_model = BTCPricePredictor.load_from_checkpoint(
    './logs/checkpoint/best-checkpoint.ckpt')
trained_model.lr

trained_model.freeze()
trained_model.eval()
test_dataset = BTCDataset(val_sequences)
predictions = []
labels = []
for item in tqdm(test_dataset):
    sequence = item['sequence']
    label = item['label']

    _, output = trained_model(sequence.unsqueeze(dim=0))
    predictions.append(output)
    labels.append(label)

# set the last elements of our trained scaler to the new descaler ==> last col is what we want to descale
de_scaler.min_, de_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]

descaled_preds = descale(de_scaler, predictions)
descaled_labels = descale(de_scaler, labels)

# plotting

test_df = data.df[data.train_size:]
test_seq_df = test_df[seq_length:]

dates = dates.date2num(test_seq_df.date.tolist())

plt.plot_date(dates, descaled_preds, '-', label='predicted')
plt.plot_date(dates, descaled_labels, '-', label='gound_truth')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('figure.png')
end = time.time()
print('Time os: ', end-start)
