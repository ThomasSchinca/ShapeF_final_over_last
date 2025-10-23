# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:35:56 2023

@author: thoma
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from functions import white_noise_dummy,extract_b_clu,Compare_RF_exo,Compare_pred_exo,Compare_RF_exo_only,Compare_pred_exo_only
from functions_deep_learning import Compare_nn_exo
import tensorflow as tf
from pmdarima.arima import auto_arima,ARIMA
np.random.seed(0)
tf.random.set_seed(1)

# Load the data from a CSV file, specifying the column names
df = pd.read_csv('Datasets/hctsa_timeseries-data.csv',names=range(10000))

# Keep only the first 1000 columns
df=df.iloc[:,:1000]

# Initialize a MinMaxScaler to scale the data
scaler = MinMaxScaler(feature_range=(0,1))

# Transpose the dataframe to switch the rows and columns
df=df.T

# Scale the data so that all values are between 0 and 1
df = scaler.fit_transform(df) 

# Transpose the dataframe again to return to the original format
df=df.T

# Convert the numpy array back to a pandas DataFrame
df=pd.DataFrame(df)

# Set up some initial parameters
names=['LSTM','LSTMX']
len_data=1000
t_spt=0.8
val_len=100

resu_nn= pd.read_csv('Results/resu_nn_up_1.csv',index_col=0)
# Iterate over 1000 rows
for row in range(len(resu_nn.columns),1000):
    try:
        # Extract the time series data for the current row
        ts_tot = df.iloc[row,:]
        ts_tot=ts_tot.dropna()
        ts = ts_tot.iloc[0:len_data]
        
        # Extract patterns from the time series 
        X = extract_b_clu(ts,[3,5,7],[3,5,7,9],train_test_split=t_spt,top=10)
        X=X.iloc[:-1,:]
        
        config = {'activation':['relu'],
                  'drop' : [0.1],
                  'l_r' : [0.001]}
        
        res_nn = Compare_nn_exo(ts,X,ar=5,number_s=5,train_test_split=t_spt,opti_grid=config)
        tot_val = pd.Series([mean_squared_error(res_nn['rf_pred'][:val_len],res_nn['obs'][:val_len]),mean_squared_error(res_nn['rfx_pred'][:val_len],res_nn['obs'][:val_len])])
        tot_res = pd.Series([mean_squared_error(res_nn['rf_pred'][val_len:],res_nn['obs'][val_len:]),mean_squared_error(res_nn['rfx_pred'][val_len:],res_nn['obs'][val_len:])])
        tot_res[len(tot_res)]=tot_res.iloc[tot_val.idxmin()]
       
        # Store the results in the appropriate dataframes
        resu_nn=pd.concat([resu_nn,tot_res],axis=1)
        print('The best model is : '+str(names[tot_val.idxmin()]))
    except:
        # If an error occurs during the model fitting process, store NaN values
        tot_res_s = pd.Series([float('NaN'),float('NaN'),float('NaN')])
        resu_nn=pd.concat([resu_nn,tot_res_s],axis=1)
    resu_nn.to_csv('Results/resu_nn_up_1.csv')
