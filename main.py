# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:36:08 2023

@author: thoma
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from functions import extract_b_clu,Compare_RF_exo,Compare_pred_exo,Compare_RF_exo_only,Compare_pred_exo_only
from functions_deep_learning import Compare_nn_exo
import random
import tensorflow as tf
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
names=['AR','ARX','RF','RFX','LSTM','LSTMX']
len_data=1000
t_spt=0.8
val_len=100

# Initialize empty dataframes to store the results
resu_4= pd.DataFrame()
resu_ar= pd.DataFrame()
resu_rf= pd.DataFrame()
resu_nn= pd.DataFrame()

# Iterate over 1000 rows
for row in range(1000):
    try:
        # Extract the time series data for the current row
        ts_tot = df.iloc[row,:]
        ts_tot=ts_tot.dropna()
        ts = ts_tot.iloc[0:len_data]
        
        # Extract patterns from the time series 
        X = extract_b_clu(ts,[3,5,7],[3,5,7,9],train_test_split=t_spt,top=10)
        X=X.iloc[:-1,:]
        
        # Initialize minimum error for ARIMA model
        min_ar_m=np.inf
        inclu =1
        # Finding the best ARIMA and ARIMA with patterns models with the minimum validation MSE
        res_temp=Compare_pred_exo(ts.iloc[4:],np.array(X.loc[3:,:].iloc[:,:inclu]),train_test_split=t_spt)
        if mean_squared_error(res_temp['Darima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len])<min_ar_m:
            res=res_temp
            min_ar_m = mean_squared_error(res_temp['Darima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len])
            res_ar=res_temp
        for inclu in [5]:
            res_temp=Compare_pred_exo_only(ts.iloc[4:],np.array(X.loc[3:,:].iloc[:,:inclu]),train_test_split=t_spt)
            if mean_squared_error(res_temp['Darima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len])<min_ar_m:
                res=res_temp
                min_ar_m = mean_squared_error(res_temp['Darima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len])
        
        # Same for RF and LSTM models
        random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               'bootstrap': [True, False]}
        config = {'activation':['relu'],
                  'n_layer' : [1],
                  'drop' : [0.1],
                  'l_r' : [0.001]}
        
        min_rf_m=np.inf
        min_rfx_m=np.inf
        inclu=1
        for ar in [1,5]:
            res_rf_temp = Compare_RF_exo(ts,X.iloc[:,:inclu],ar=ar,number_s=3,train_test_split=t_spt,opti_grid=random_grid)
            if mean_squared_error(res_rf_temp['rf_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])<min_rf_m:
                res_rf=res_rf_temp
                min_rf_m = mean_squared_error(res_rf_temp['rf_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])
            if mean_squared_error(res_rf_temp['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])<min_rfx_m:
                res_rfx=res_rf_temp
                min_rfx_m = mean_squared_error(res_rf_temp['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])    
            for inclu in [5]:
                res_rf_temp = Compare_RF_exo_only(ts,X.iloc[:,:inclu],ar=ar,number_s=3,train_test_split=t_spt,opti_grid=random_grid)
                if mean_squared_error(res_rf_temp['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])<min_rfx_m:
                    res_rfx=res_rf_temp
                    min_rfx_m = mean_squared_error(res_rf_temp['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])
                    
        res_nn = Compare_nn_exo(ts,X,ar=5,number_s=5,train_test_split=t_spt,opti_grid=config)
        tot_val = pd.Series([mean_squared_error(res_ar['arima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len]),mean_squared_error(res['Darima_pred'][:val_len],ts[int(t_spt*len(ts)):][:val_len]),mean_squared_error(res_rf['rf_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len]),mean_squared_error(res_rfx['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len]),mean_squared_error(res_nn['rf_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len]),mean_squared_error(res_nn['rfx_pred'][:val_len],ts[int(t_spt*len(ts))-1:-1][:val_len])])
        tot_res = pd.Series([mean_squared_error(res_ar['arima_pred'][val_len:],ts[int(t_spt*len(ts)):][val_len:]),mean_squared_error(res['Darima_pred'][val_len:],ts[int(t_spt*len(ts)):][val_len:]),mean_squared_error(res_rf['rf_pred'][val_len:],ts[int(t_spt*len(ts))-1:-1][val_len:]),mean_squared_error(res_rfx['rfx_pred'][val_len:],ts[int(t_spt*len(ts))-1:-1][val_len:]),mean_squared_error(res_nn['rf_pred'][val_len:],ts[int(t_spt*len(ts))-1:-1][val_len:]),mean_squared_error(res_nn['rfx_pred'][val_len:],ts[int(t_spt*len(ts))-1:-1][val_len:])])
        tot_res[len(tot_res)]=tot_res.iloc[tot_val.idxmin()]
        tot_ar = tot_res.iloc[:2]
        tot_ar[2] = tot_res.iloc[tot_val.iloc[:2].idxmin()]
        tot_rf = tot_res.iloc[2:4]
        tot_rf[4] = tot_res.iloc[tot_val.iloc[2:4].idxmin()]
        tot_nn = tot_res.iloc[4:6]
        tot_nn[6] = tot_res.iloc[tot_val.iloc[4:6].idxmin()]
        
        # Store the results in the appropriate dataframes
        resu_4=pd.concat([resu_4,tot_res],axis=1)
        resu_rf=pd.concat([resu_rf,tot_rf],axis=1)
        resu_nn=pd.concat([resu_nn,tot_nn],axis=1)
        resu_ar=pd.concat([resu_ar,tot_ar],axis=1)
        print('The best model is : '+str(names[tot_val.idxmin()]))
        
    except:
        # If an error occurs during the model fitting process, store NaN values
        tot_res = pd.Series([float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')])
        tot_res_s = pd.Series([float('NaN'),float('NaN'),float('NaN')])
        resu_4=pd.concat([resu_4,tot_res],axis=1)
        resu_rf=pd.concat([resu_rf,tot_res_s],axis=1)
        resu_nn=pd.concat([resu_nn,tot_res_s],axis=1)
        resu_ar=pd.concat([resu_ar,tot_res_s],axis=1)
    # Write the results to CSV files    
    resu_4.to_csv('Results/resu_tot.csv')
    resu_ar.to_csv('Results/resu_ar.csv')
    resu_rf.to_csv('Results/resu_rf.csv')
    resu_nn.to_csv('Results/resu_nn.csv')











