import pandas as pd
import numpy as np
import pickle 

n_rows = 100000

dataset = pd.DataFrame(
    data={
        'string': np.random.choice(('apple', 'banana', 'carrot'), size=n_rows),
        'timestamp': pd.date_range("20130101", periods=n_rows, freq="s"),
        'integer': np.random.choice(range(0,10), size=n_rows),
        'float': np.random.uniform(size=n_rows),
    },
)

dataset.to_csv('./assets/data.csv', index = False)
dataset.to_feather('./assets/data.feather')
with open('./assets/data.pickle', 'wb') as file:
    pickle.dump(dataset, file)

dataset.to_parquet('./assets/data.parquet')
# dataset.to_hdf('./assets/data.h5', key='dataset', mode='w')
dataset.to_json('./assets/data.json')
dataset.to_xarray().to_netcdf('dataset.nc', engine='h5netcdf')