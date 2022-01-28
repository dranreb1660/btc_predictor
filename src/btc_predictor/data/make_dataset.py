

from fastbook import *
import urllib.request
import ssl

base_url = './'

ssl._create_default_https_context = ssl._create_unverified_context

path = Path('./')

url = "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv"
path = download_url(url, dest=path)
