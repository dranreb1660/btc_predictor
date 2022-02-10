import ssl
import os
from sklearn.preprocessing import MinMaxScaler
from fastbook import *
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


ssl._create_default_https_context = ssl._create_unverified_context
base_path = os.path.abspath('.')
print('abs path is ', base_path)
data_url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv"
# download the data with url to data/raw folder


class Features():
    def __init__(self, data_url=data_url) -> None:
        self.data_url = data_url
        self.raw_output_path = os.path.join(
            str(base_path) + '/data/raw/')
        self.interim_output_path = os.path.join(
            str(base_path) + '/data/interim/')
        self.processed_output_path = os.path.join(
            str(base_path) + '/data/processed/')
        self.scaler = None
        self.train_df, self.val_df = None, None

    def _download_data(self):
        """download data from given url with fatsai download_url method"""
        print('\n......Dounloading data from url ........\n')
        self.raw_data = download_url(
            self.data_url, dest=Path(self.raw_output_path))

        print('')
        print('raw_data--', self.raw_data)
        return self.raw_data

    def build_features(self):
        # read
        raw_data_path = str(self._download_data())
        df = pd.read_csv(raw_data_path, parse_dates=[
                         'date'], low_memory=False, header=1)
        print(df.shape)
        df = df.sort_values(by='date').reset_index(drop=True)  # sort by date

        # taking a small sample
        df = df[df.date >= '2021-12-15 02:33:00']
        df = df.reset_index(drop=True)

        # shift the target vars by one to get the previos days close
        df['prev_close'] = df.shift(1)['close']
        print(df.shape)
        # creating a new column which will be the closing chng price = close - prev_close
        df['close_change'] = df.progress_apply(
            lambda row: 0 if np.isnan(row.prev_close) else row.close - row.prev_close, axis=1
        )
        print('pp', df.shape)
        # Building and selecting features
        print('\n......Building selected features.........\n')
        rows = []
        ks = []
        for k, row in tqdm(df.iterrows(), total=df.shape[0]):
            row_data = dict(
                day_of_week=row.date.dayofweek,
                day_of_month=row.date.day,
                week_of_year=row.date.weekofyear,
                month=row.date.month,
                open=row.open,
                high=row.high,
                low=row.low,
                close_change=row.close_change,
                close=row.close
            )
            rows.append(row_data)
        self.features_df = pd.DataFrame(rows)
        self.features_df.to_csv(self.interim_output_path+'features.csv')

        print('\n......Building selected features Complete.........\n')
        return self.features_df

    def _split(self, train_ratio: float = .9):
        feats = self.build_features()
        train_size = int(len(feats) * train_ratio)
        self.train_df, self.val_df = self.features_df[:
                                                      train_size], self.features_df[train_size:]
        return self.train_df, self.val_df

    def get_train_val_scaled(self, scaler=MinMaxScaler(feature_range=(-1, 1))):
        train, val = self._split()
        self.scaler = scaler
        self.scaler = self.scaler.fit(train)

        scaled_train_df = self.scale_df(train)
        scaled_val_df = self.scale_df(val)
        scaled_train_df.to_csv(self.processed_output_path+'scaled_train.csv'),
        scaled_val_df.to_csv(self.processed_output_path+'scaled_valid.csv')
        return scaled_train_df, scaled_val_df

    def scale_df(self, df):
        df = pd.DataFrame(self.scaler.transform(df),
                          index=df.index,
                          columns=df.columns)

        return df
