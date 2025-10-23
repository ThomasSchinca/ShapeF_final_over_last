# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:43:10 2023

@author: thoma
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel,ttest_1samp
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from functions import get_dynamic_clusters
from scipy.cluster.hierarchy import dendrogram, linkage
from tslearn.clustering import TimeSeriesKMeans
import random
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D 

# Load ARIMA result data, drop missing values, and calculate the mean of each column
df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot = df_tot.dropna(axis=0)
df_tot.columns=['AR','ARX','ARC']
print(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]), 0))
print(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])*df_tot.iloc[:,0], 0))


df_tot_0= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot_0=df_tot_0.T
df_tot_0=df_tot_0.reset_index(drop=True)
df_tot_0 = df_tot_0.dropna(axis=1,how='all')
df_tot_0 = df_tot_0.dropna(axis=0)
df_tot_0.columns=['LSTM','LSTMX','LSTMC']
means_nn = df_tot_0.mean(axis=0).sort_values(ascending=False)
print(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])*df_tot_0.iloc[:,0], 0))


df_tot_1= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1 = df_tot_1.iloc[:,:3]
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_1 = df_tot_1.dropna(axis=0)
df_tot_1.columns=['RF','RFX','RFC']
means_rf = df_tot_1.mean(axis=0).sort_values(ascending=False)
print(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])*df_tot_1.iloc[:,0], 0))

# =============================================================================
# Features
# =============================================================================

# Load the feature datasets
t_cara = pd.read_csv('Datasets/hctsa_features.csv',index_col=(0))
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
data_m = pd.read_csv('Datasets/hctsa_datamatrix.csv',header=None)

# Analyze the AR model
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:, -1]  # Select the last column as y
X = data_c.iloc[:, :-1]  # Select all other columns as X

# Find the columns with a correlation higher than 0.8 with y
significant_vars = []
for col in X.columns:
    corr = X[col].corr(y)
    if abs(corr) > 0.8:
        significant_vars.append(col)

# Select only the significant variables, add a constant column, and drop missing values
X =X.iloc[:,significant_vars]
X = sm.add_constant(X)
X = X.dropna()

# Select the corresponding y values
y_na = y.loc[X.index]

# Fit an OLS model and print the summary
model = sm.OLS(y_na,X).fit()
p_values = model.pvalues
print(model.summary())

# Repeat the same steps for RF and LSTM models
# RF

data_c = pd.concat([data_m,np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:, -1]  # Select the last column as y
X = data_c.iloc[:, :-1]  # Select all other columns as X
significant_vars = []
for col in X.columns:
    corr = X[col].corr(y)
    
    # If the correlation is higher than 0.8, add the column to the significant_vars list
    if abs(corr) > 0.8:
        significant_vars.append(col)
        
X =X.iloc[:,significant_vars]
X = sm.add_constant(X)
X = X.dropna()
y_na = y.loc[X.index]
model = sm.OLS(y_na,X).fit()
p_values = model.pvalues
print(model.summary())

# LSTM

data_c = pd.concat([data_m,np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:, -1]  # Select the last column as y
X = data_c.iloc[:, :-1]  # Select all other columns as X
significant_vars = []
for col in X.columns:
    corr = X[col].corr(y)
    
    # If the correlation is higher than 0.8, add the column to the significant_vars list
    if abs(corr) > 0.8:
        significant_vars.append(col)
        
X =X.iloc[:,significant_vars]
X = sm.add_constant(X)
X = X.dropna()
y_na = y.loc[X.index]
model = sm.OLS(y_na,X).fit()
p_values = model.pvalues
print(model.summary())

# =============================================================================
# Type of TS
# =============================================================================

# Extract and parse keywords from 'Keywords' column of cara_df dataframe
ts_kind=[]
for i in cara_df['Keywords']:
    text = i.split(',')
    if text[0]=='synthetic':
        if (text[1]=='map') or (text[1]=='dynsys'):  
            ts_kind.append(text[2])
        else:    
            ts_kind.append(text[1])
    else : 
        ts_kind.append(text[0])

# Generate a dataframe with log-ratios of RF model's MSE and filter valid rows
data_c = pd.concat([data_m,np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]

# Combine the parsed keywords and log-ratios into a dataframe
ts_kind_t = pd.concat([pd.Series(ts_kind),data_c.iloc[:,-1]],axis=1)
ts_kind_t = ts_kind_t.dropna(axis=0)
ts_kind_t.columns=['Type','Ratio']

# Group the dataframe by 'Type' and calculate the mean, then sort the dataframe
ts_kind_tot = ts_kind_t.groupby(['Type']).mean()
ts_kind_tot_rf= ts_kind_tot.sort_values(by=['Ratio'],ascending=True)

# Plot the sorted dataframe as horizontal bar plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(65, 30))

# RF Model
ts_kind_tot_rf.plot(kind='barh', ax=ax1)
ax1.set_title('Mean MSE Ratio - RF')
ax1.set_ylabel('Fields of the TS')
ax1.set_xlabel('')   
ax1.set_xlim(-max(abs(ts_kind_tot_rf.values)), max(abs(ts_kind_tot_rf.values)))

# Repeat the same process for LSTM and AR models
# LSTM Model
data_c = pd.concat([data_m,np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
ts_kind_t = pd.concat([pd.Series(ts_kind),data_c.iloc[:,-1]],axis=1)
ts_kind_t = ts_kind_t.dropna(axis=0)
ts_kind_t.columns=['Type','Ratio']
ts_kind_tot = ts_kind_t.groupby(['Type']).mean()
ts_kind_tot_lstm= ts_kind_tot.sort_values(by=['Ratio'],ascending=True)
ts_kind_tot_lstm.plot(kind='barh', ax=ax2)

ax2.set_xlabel('Log Ratio')
ax2.set_title('Mean MSE Ratio - LSTM') 
ax2.set_ylabel('')       
ax2.set_xlim(-max(abs(ts_kind_tot_lstm.values)), max(abs(ts_kind_tot_lstm.values)))


# AR Model
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
ts_kind_t = pd.concat([pd.Series(ts_kind),data_c.iloc[:,-1]],axis=1)
ts_kind_t = ts_kind_t.dropna(axis=0)
ts_kind_t.columns=['Type','Ratio']
ts_kind_tot = ts_kind_t.groupby(['Type']).mean()
ts_kind_tot_ar= ts_kind_tot.sort_values(by=['Ratio'],ascending=True)
ts_kind_tot_ar.plot(kind='barh', ax=ax3)
ax3.set_ylabel('')
ax3.set_title('Mean MSE Ratio - AR')
ax3.set_xlim(-max(abs(ts_kind_tot_ar.values)), max(abs(ts_kind_tot_ar.values)))
positive_ar = ts_kind_tot_ar[ts_kind_tot_ar > 0].dropna()
positive_nn = ts_kind_tot_lstm[ts_kind_tot_lstm > 0].dropna()
positive_rf = ts_kind_tot_rf[ts_kind_tot_rf > 0].dropna()
common_indexes = set(positive_rf.index) & set(positive_ar.index) & set(positive_nn.index)
indexes_rf = list(ts_kind_tot_rf.index)
indexes_ar = list(ts_kind_tot_ar.index)
indexes_nn = list(ts_kind_tot_lstm.index)
def highlight_common_positives(indexes, ax):
    for i, bar in enumerate(ax.patches):
        y_label = indexes[i]  # Get the index corresponding to the current bar
        if y_label in common_indexes and bar.get_width() > 0:
            bar.set_facecolor('red')
            ax.get_yticklabels()[i].set_color('red')

# Highlight common positive indexes in red for all three plots
highlight_common_positives(indexes_rf, ax1)
highlight_common_positives(indexes_nn, ax2)
highlight_common_positives(indexes_ar, ax3)

for ax in (ax1, ax2, ax3):
    ax.legend_.remove()
plt.show()




# tot_kind = pd.concat([ts_kind_tot_ar,ts_kind_tot_rf,ts_kind_tot_lstm],axis=1)
# tot_kind= tot_kind.dropna()
# tot_kind.columns = ['AR','RF','LSTM']

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for name, row in tot_kind.iterrows():
#     x, y, z = row
#     color = 'red' if all(row > 0) else 'blue'
#     ax.scatter(x, y, z, c=color, label=name)

#     if color == 'red':
#         ax.text(x, y, z, name, color='red')
#     else:
#         ax.text(x, y, z, name, color='blue')
# ax.set_xlabel('AR')
# ax.set_ylabel('RF')
# ax.set_zlabel('LSTM')
# plt.show()


# =============================================================================
# Cluster exemple 
# =============================================================================

# df = pd.read_csv('Datasets/hctsa_timeseries-data.csv',names=range(10000))
# df=df.iloc[:,:1000]
# scaler = MinMaxScaler(feature_range=(0,1))
# df=df.T
# df = scaler.fit_transform(df)
# df=df.T
# df=pd.DataFrame(df)
# colo={'C0':'tab:blue','C1':'tab:orange','C2':'tab:green','C3':'tab:red','C4':'tab:purple'}

# h= get_dynamic_clusters(df.iloc[85,:],n_clu=9,number_s=5)
# h['cluster_shape']=h['cluster_shape'].reshape((9,5))
# Z = linkage(h['cluster_shape'])
# fig, ax = plt.subplots(figsize=(20, 6))
# k=dendrogram(Z)
# ku= list(map(int,k['ivl']))
# ku_col=k['leaves_color_list']
# dendrogram(Z, orientation='top', ax=ax)
# co=0
# for i, centroid in enumerate(h['cluster_shape']):
#     centroid_ax = ax.inset_axes([(ku.index(co)/9), -0.45, 1/9, 0.4])
#     centroid_ax.plot(centroid, color=colo[ku_col[ku.index(co)]])
#     centroid_ax.axis('off')
#     centroid_ax.set_ylim(0,1)
#     for k in h['sequences'][h['seqences_clusters']==i]:
#         centroid_ax.plot(k, color=colo[ku_col[ku.index(co)]],alpha=0.05)
#     co=co+1
# plt.xticks([*range(9)],['']*9)
# #plt.title(hu)
# plt.axis('off')
# plt.yticks([0],[''])
# plt.show()


# # =============================================================================
# # MSE Bootstrap
# # =============================================================================


# glob = pd.concat([df_tot,df_tot_1,df_tot_0],axis=1)
# glob.columns=[0,1,2,4,5,6,8,9,10]
# mean=[]
# std=[]
# for i in range(11):
#     if i in [3,7]:
#         mean.append(float('NaN'))
#         std.append(float('NaN'))
#     else:    
#         seq=[]
#         for rep in range(1000):
#             sample = random.choices(glob.loc[:,i].tolist(),k=840)
#             seq.append(pd.Series(sample).mean())
#         seq=pd.Series(seq)    
#         mean.append(seq.mean())
#         std.append(seq.std())
    

# # Calculate the confidence interval for the difference in MSEs
# mean2=[]
# mean2.append(float('NaN'))
# mean2.append((df_tot.iloc[:,0]-df_tot.iloc[:,1]).mean())
# mean2.append((df_tot.iloc[:,0]-df_tot.iloc[:,2]).mean())
# mean2.append(float('NaN'))
# mean2.append(float('NaN'))
# mean2.append((df_tot_1.iloc[:,0]-df_tot_1.iloc[:,1]).mean())
# mean2.append((df_tot_1.iloc[:,0]-df_tot_1.iloc[:,2]).mean())
# mean2.append(float('NaN'))
# mean2.append(float('NaN'))
# mean2.append((df_tot_0.iloc[:,0]-df_tot_0.iloc[:,1]).mean())
# mean2.append((df_tot_0.iloc[:,0]-df_tot_0.iloc[:,2]).mean())
# std2=[]
# std2.append(float('NaN'))
# std2.append(2*(df_tot.iloc[:,0]-df_tot.iloc[:,1]).std()/np.sqrt(len(df_tot)))
# std2.append(2*(df_tot.iloc[:,0]-df_tot.iloc[:,2]).std()/np.sqrt(len(df_tot)))
# std2.append(float('NaN'))
# std2.append(float('NaN'))
# std2.append(2*(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,1]).std()/np.sqrt(len(df_tot_1)))
# std2.append(2*(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,2]).std()/np.sqrt(len(df_tot_1)))
# std2.append(float('NaN'))
# std2.append(float('NaN'))
# std2.append(2*(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,1]).std()/np.sqrt(len(df_tot_0)))
# std2.append(2*(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,2]).std()/np.sqrt(len(df_tot_0)))


# # Create a DataFrame for mean2 and std2
# mean2_data = pd.DataFrame({
#     'mean2': mean2,
#     'std2': std2
# })

# # Create a DataFrame for mean and std
# mean_data = pd.DataFrame({
#     'mean': mean,
#     'std': std
# })

# blue_color = '#404040'  # Dark grey shade
# orange_color = '#A0A0A0'  # Light grey shade

# def mse_formatter(x, pos):
#     return '{:.1e}'.format(x)
# def improvement_formatter(x, pos):
#     return '{:.1e}'.format(x)

# # Create the figure and axes
# fig, ax1 = plt.subplots(figsize=(12,8))

# # Increase marker size and linewidth
# marker_size = 150
# linewidth = 3
# fonts=25
# # Set larger font size for text
# plt.rc('font', size=24)

# # Set y-axis tick colors
# ax1.yaxis.label.set_color(blue_color)
# ax1.tick_params(axis='y', colors=blue_color)

# # Set y-axis formatter and plot data for ax1
# ax1.yaxis.set_major_formatter(FuncFormatter(mse_formatter))
# ax1.scatter(mean_data.index, mean_data['mean'], color=blue_color, marker='o', s=marker_size)
# ax1.errorbar(mean_data.index, mean_data['mean'], yerr=mean_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
# # ax1.set_ylim(-0.0038, 0.015)


# # Create a second y-axis for mean2
# ax2 = ax1.twinx()
# ax2.grid(False)

# # Set y-axis tick colors for ax2
# ax2.yaxis.label.set_color(orange_color)
# ax2.tick_params(axis='y', colors=orange_color)

# # Set y-axis formatter and plot data for ax2
# ax2.yaxis.set_major_formatter(FuncFormatter(improvement_formatter))
# ax2.scatter(mean2_data.index, mean2_data['mean2'], color=orange_color, marker='o', s=marker_size)
# ax2.errorbar(mean2_data.index, mean2_data['mean2'], yerr=mean2_data['std2'], fmt='none', color=orange_color, linewidth=linewidth)
# ax2.hlines(0, -0.5, 10.5, linestyles='--', color=orange_color, linewidth=linewidth)
# ax2.set_ylim(-0.0004, 0.002)

# # Set labels and title
# ax1.set_xlabel('Models',fontsize=fonts)
# ax1.set_ylabel('MSE', color=blue_color,fontsize=fonts)
# ax2.set_ylabel('MSE Improvement', color=orange_color,fontsize=fonts)

# ax1.set_xticks([*range(11)],['AR','ARX','ARC','','RF','RFX','RFC','','LSTM','LSTMX','LSTMC'],fontsize=fonts,rotation=45)
# ax1.set_yticks([0,0.005,0.01,0.017],['0','0.005','0.01','0.015'],fontsize=fonts)
# ax1.set_yticks([],fontsize=fonts)
# ax2.set_yticks([0],['0'],fontsize=fonts)
# ax1.grid(False)
# plt.show()


# =============================================================================
# MSE Bootstrap
# =============================================================================


glob = pd.concat([df_tot,df_tot_1,df_tot_0],axis=1)
glob.columns=[0,1,2,4,5,6,8,9,10]
mean=[]
std=[]
for i in range(11):
    if i in [3,7]:
        mean.append(float('NaN'))
        std.append(float('NaN'))
    else:    
        seq=[]
        for rep in range(1000):
            sample = random.choices(glob.loc[:,i].tolist(),k=840)
            seq.append(pd.Series(sample).mean())
        seq=pd.Series(seq)    
        mean.append(seq.mean())
        std.append(seq.std())
    

# Calculate the confidence interval for the difference in MSEs
mean2=[]
mean2.append(float('NaN'))
mean2.append(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).mean())
mean2.append(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).mean())
mean2.append(float('NaN'))
mean2.append(float('NaN'))
mean2.append(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).mean())
mean2.append(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).mean())
mean2.append(float('NaN'))
mean2.append(float('NaN'))
mean2.append(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).mean())
mean2.append(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).mean())
std2=[]
std2.append(float('NaN'))
std2.append(2*np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).std()/np.sqrt(len(df_tot)))
std2.append(2*np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).std()/np.sqrt(len(df_tot)))
std2.append(float('NaN'))
std2.append(float('NaN'))
std2.append(2*np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).std()/np.sqrt(len(df_tot_1)))
std2.append(2*np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).std()/np.sqrt(len(df_tot_1)))
std2.append(float('NaN'))
std2.append(float('NaN'))
std2.append(2*np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).std()/np.sqrt(len(df_tot_0)))
std2.append(2*np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).std()/np.sqrt(len(df_tot_0)))

# Create a DataFrame for mean2 and std2
mean2_data = pd.DataFrame({
    'mean2': mean2,
    'std2': std2
})

# Create a DataFrame for mean and std
mean_data = pd.DataFrame({
    'mean': mean,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade

def mse_formatter(x, pos):
    return '{:.1e}'.format(x)
def improvement_formatter(x, pos):
    return '{:.1e}'.format(x)

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(12,8))

# Increase marker size and linewidth
marker_size = 150
linewidth = 3
fonts=25
# Set larger font size for text
plt.rc('font', size=24)

# Set y-axis tick colors
ax1.yaxis.label.set_color(blue_color)
ax1.tick_params(axis='y', colors=blue_color)

# Set y-axis formatter and plot data for ax1
ax1.yaxis.set_major_formatter(FuncFormatter(mse_formatter))
ax1.scatter(mean_data.index, mean_data['mean'], color=blue_color, marker='o', s=marker_size)
ax1.errorbar(mean_data.index, mean_data['mean'], yerr=mean_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
# ax1.set_ylim(-0.0038, 0.015)


# Create a second y-axis for mean2
ax2 = ax1.twinx()
ax2.grid(False)

# Set y-axis tick colors for ax2
ax2.yaxis.label.set_color(orange_color)
ax2.tick_params(axis='y', colors=orange_color)

# Set y-axis formatter and plot data for ax2
ax2.yaxis.set_major_formatter(FuncFormatter(improvement_formatter))
ax2.scatter(mean2_data.index, mean2_data['mean2'], color=orange_color, marker='o', s=marker_size)
ax2.errorbar(mean2_data.index, mean2_data['mean2'], yerr=mean2_data['std2'], fmt='none', color=orange_color, linewidth=linewidth)
ax2.hlines(0, -0.5, 10.5, linestyles='--', color=orange_color, linewidth=linewidth)

# Set labels and title
ax1.set_xlabel('Models',fontsize=fonts)
ax1.set_ylabel('MSE', color=blue_color,fontsize=fonts)
ax2.set_ylabel('MSE Improvement', color=orange_color,fontsize=fonts)

ax1.set_xticks([*range(11)],['AR','ARX','ARC','','RF','RFX','RFC','','LSTM','LSTMX','LSTMC'],fontsize=fonts,rotation=45)
ax1.set_yticks([0,0.005,0.01,0.017],['0','0.005','0.01','0.015'],fontsize=fonts)
ax1.set_yticks([],fontsize=fonts)
ax2.set_yticks([0],['0'],fontsize=fonts)
ax1.grid(False)
plt.show()


# =============================================================================
# New metric
# =============================================================================

mean2=[]
mean2.append(float('NaN'))
mean2.append((np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1])*df_tot.iloc[:,0]).mean())
mean2.append((np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])*df_tot.iloc[:,0]).mean())
mean2.append(float('NaN'))
mean2.append(float('NaN'))
mean2.append((np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1])*df_tot_1.iloc[:,0]).mean())
mean2.append((np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])*df_tot_1.iloc[:,0]).mean())
mean2.append(float('NaN'))
mean2.append(float('NaN'))
mean2.append((np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1])*df_tot_0.iloc[:,0]).mean())
mean2.append((np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])*df_tot_0.iloc[:,0]).mean())
std2=[]
std2.append(float('NaN'))
std2.append((2*np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1])*df_tot.iloc[:,0]).std()/np.sqrt(len(df_tot)))
std2.append((2*np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])*df_tot.iloc[:,0]).std()/np.sqrt(len(df_tot)))
std2.append(float('NaN'))
std2.append(float('NaN'))
std2.append((2*np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1])*df_tot_1.iloc[:,0]).std()/np.sqrt(len(df_tot_1)))
std2.append((2*np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])*df_tot_1.iloc[:,0]).std()/np.sqrt(len(df_tot_1)))
std2.append(float('NaN'))
std2.append(float('NaN'))
std2.append((2*np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1])*df_tot_0.iloc[:,0]).std()/np.sqrt(len(df_tot_0)))
std2.append((2*np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])*df_tot_0.iloc[:,0]).std()/np.sqrt(len(df_tot_0)))

# Create a DataFrame for mean2 and std2
mean2_data = pd.DataFrame({
    'mean2': mean2,
    'std2': std2
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade

def mse_formatter(x, pos):
    return '{:.1e}'.format(x)
def improvement_formatter(x, pos):
    return '{:.1e}'.format(x)

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(12,8))

# Increase marker size and linewidth
marker_size = 150
linewidth = 3
fonts=25
# Set larger font size for text
plt.rc('font', size=24)

# Set y-axis tick colors
ax1.yaxis.label.set_color(blue_color)
ax1.tick_params(axis='y', colors=blue_color)

# Set y-axis formatter and plot data for ax1
ax1.yaxis.set_major_formatter(FuncFormatter(mse_formatter))
ax1.scatter(mean2_data.index, mean2_data['mean2'], color=blue_color, marker='o', s=marker_size)
ax1.errorbar(mean2_data.index, mean2_data['mean2'], yerr=mean2_data['std2'], fmt='none', color=blue_color, linewidth=linewidth)

# Set labels and title
ax1.set_xlabel('Models',fontsize=fonts)
ax1.set_ylabel('MSE', color=blue_color,fontsize=fonts)
ax2.set_ylabel('MSE Improvement', color=orange_color,fontsize=fonts)

ax1.set_xticks([*range(11)],['AR','ARX','ARC','','RF','RFX','RFC','','LSTM','LSTMX','LSTMC'],fontsize=fonts,rotation=45)
ax1.set_ylim(-0.001,0.009)
ax1.hlines(0,-0.5,12,linestyle='--')
ax1.grid(False)
ax1.set_title('Real')
plt.show()


# =============================================================================
# Plot Improv
# =============================================================================

# plt.plot(df_tot.iloc[:,0]-df_tot.iloc[:,1])
# plt.title('AR-ARX')
# plt.show()

# plt.plot(df_tot.iloc[:,0]-df_tot.iloc[:,2])
# plt.title('AR-ARC')
# plt.show()


# plt.plot(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,1])
# plt.title('RF-RFX')
# plt.show()

# plt.plot(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,2])
# plt.title('RF-RFC')
# plt.show()

# plt.plot(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,1])
# plt.title('NN-NNX')
# plt.show()

# plt.plot(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,2])
# plt.title('NN-NNC')
# plt.show()





# =============================================================================
# Incluson of clusters
# =============================================================================

df_tot_inc=pd.DataFrame(index=['RF','RFX','RFC'])
df_tot_inc_log_rat = pd.DataFrame()
for i in [1,5,10,15,20]:
    df_tot_1= pd.read_csv(f'Results/resu_{i}.csv',index_col=(0))
    df_tot_1=df_tot_1.T
    df_tot_1=df_tot_1.reset_index(drop=True)
    df_tot_1 = df_tot_1.dropna(axis=0)
    df_tot_1.columns=['RF','RFX','RFC']
    df_tot_inc = pd.concat([df_tot_inc,df_tot_1.mean()],axis=1)
    df_tot_inc_log_rat= pd.concat([df_tot_inc_log_rat,pd.Series([np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).mean(),np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).mean()])],axis=1)
df_tot_inc.columns=[1,10,20,30,59]
df_tot_inc_log_rat.columns=[1,10,20,30,59]



plt.figure(figsize=(10,6))
plt.plot(df_tot_inc.iloc[0,:],label='RF',marker='o')
plt.plot(df_tot_inc.iloc[1,:],label='RFX',marker='o')
plt.plot(df_tot_inc.iloc[2,:],label='RFC',marker='o')
plt.legend()
plt.xlabel('Nb cluster included')
plt.ylabel("MSE")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df_tot_inc_log_rat.iloc[0,:],label='RFX',marker='o')
plt.plot(df_tot_inc_log_rat.iloc[1,:],label='RFC',marker='o')
plt.legend()
plt.xlabel('Nb cluster included')
plt.ylabel("Log ratio with MSE RF")
plt.show()

df_stds =  pd.read_csv('Results/stds.csv',index_col=(0))
df_stds=df_stds.T
df_stds = df_stds.reset_index(drop=True)


df_selec_ar = np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])
df_selec_ar=df_selec_ar[df_selec_ar!=0]
df_selec_rf = np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])
df_selec_rf=df_selec_rf[df_selec_rf!=0]
df_selec_nn = np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])
df_selec_nn=df_selec_nn[df_selec_nn!=0]
df_stds_err = pd.concat([df_stds[1],df_selec_ar,df_selec_rf,df_selec_nn],axis=1) 
df_stds_err = df_stds_err.dropna()
ar_coef = np.polyfit(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 1], 1)
ar_poly = np.poly1d(ar_coef)
rf_coef = np.polyfit(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 2], 1)
rf_poly = np.poly1d(rf_coef)
lstm_coef = np.polyfit(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 3], 1)
lstm_poly = np.poly1d(lstm_coef)


plt.figure(figsize=(10,6))
plt.scatter(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 1], label='AR')
plt.plot(df_stds_err.iloc[:, 0], ar_poly(df_stds_err.iloc[:, 0]), label='AR Regression Line', linestyle='--')
plt.title('Arima')
plt.ylabel('Log ratio MSE(Auto)/MSE(cluster)')
plt.xlabel('Mean std of the clusters')
plt.grid()
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 2], label='RF')
plt.scatter(df_stds_err.iloc[:, 0], df_stds_err.iloc[:, 3], label='LSTM')
plt.plot(df_stds_err.iloc[:, 0], rf_poly(df_stds_err.iloc[:, 0]))
plt.plot(df_stds_err.iloc[:, 0], lstm_poly(df_stds_err.iloc[:, 0]))
plt.legend()
plt.ylabel('Log ratio MSE(Auto)/MSE(cluster)')
plt.xlabel('Mean std of the clusters')
plt.grid()
plt.show()



df_selec_ar = np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])
df_selec_ar_nz=df_selec_ar[df_selec_ar!=0]
df_selec_ar_z=df_selec_ar[df_selec_ar==0]
df_selec_rf = np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2])
df_selec_rf_nz=df_selec_rf[df_selec_rf!=0]
df_selec_rf_z=df_selec_rf[df_selec_rf==0]
df_selec_nn = np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2])
df_selec_nn_nz=df_selec_nn[df_selec_nn!=0]
df_selec_nn_z=df_selec_nn[df_selec_nn==0]

df_stds_err_nz = pd.concat([df_stds[2],df_selec_ar_nz,df_selec_rf_nz,df_selec_nn_nz],axis=1) 
df_stds_err_z = pd.concat([df_stds[2],df_selec_ar_z,df_selec_rf_z,df_selec_nn_z],axis=1) 


bins = np.linspace(0,0.32,41)
for i in range(1,4):
    boxp = df_stds_err_z.iloc[:,[0,i]].dropna()
    boxp_c = df_stds_err_nz.iloc[:,[0,i]].dropna()
    plt.figure(figsize=(10,6))
    plt.hist(boxp.iloc[:,0], bins=bins,alpha=0.5,label='AutoReg model',color='orange')
    plt.hist(boxp_c.iloc[:,0], bins=bins,alpha=0.5,label='Cluster model',color='blue')
    plt.vlines(boxp.iloc[:,0].mean(),0,35,color='orange',linestyles='--')
    plt.vlines(boxp_c.iloc[:,0].mean(),0,35,color='blue',linestyles='--')
    plt.xlabel('Mean std of the clusters')
    plt.legend()
    
    plt.show()
 

boxp = df_stds_err_z.dropna()
boxp_c = df_stds_err_nz.dropna()
plt.figure(figsize=(10,6))
plt.hist(boxp.iloc[:,0], bins=bins,alpha=0.5,label='AutoReg model',color='orange')
plt.hist(boxp_c.iloc[:,0], bins=bins,alpha=0.5,label='Cluster model',color='blue')
plt.vlines(boxp.iloc[:,0].mean(),0,14,color='orange',linestyles='--')
plt.vlines(boxp_c.iloc[:,0].mean(),0,14,color='blue',linestyles='--')
plt.legend()
plt.xlabel('Mean std of the clusters')
plt.show()   





def calculate_counts(df):
    log_ratios_tot = np.log(df.iloc[:, 0] / df.iloc[:, 1])
    neg_count_tot = np.sum(log_ratios_tot < 0)
    zero_count_tot = np.sum(log_ratios_tot == 0)
    pos_count_tot = np.sum(log_ratios_tot > 0)

    log_ratios_tot_r = np.log(df.iloc[:, 0] / df.iloc[:, 2])
    neg_count_tot_r = np.sum(log_ratios_tot_r < 0)
    zero_count_tot_r = np.sum(log_ratios_tot_r == 0)
    pos_count_tot_r = np.sum(log_ratios_tot_r > 0)

    neg_count_diff = neg_count_tot_r - neg_count_tot
    zero_count_diff = zero_count_tot_r - zero_count_tot
    pos_count_diff = pos_count_tot_r - pos_count_tot

    categories = ['Negative', 'Zero', 'Positive']
    counts_tot = [neg_count_tot, zero_count_tot, pos_count_tot]
    counts_tot_r = [neg_count_tot_r, zero_count_tot_r, pos_count_tot_r]
    counts_diff = [neg_count_diff, zero_count_diff, pos_count_diff]

    counts_tot = counts_tot / sum(counts_tot) * 100
    counts_tot_r = counts_tot_r / sum(counts_tot_r) * 100
    counts_diff = counts_tot_r - counts_tot

    return categories, counts_tot, counts_tot_r, counts_diff


df_tot_r = pd.read_csv('Results/resu_ar.csv', index_col=(0)).T.dropna(axis=0).reset_index(drop=True)
df_tot_r.columns = ['AR', 'ARX', 'ARC']

df_tot_0_r = pd.read_csv('Results/resu_nn.csv', index_col=(0)).T.dropna(axis=1, how='all').dropna(axis=0).reset_index(drop=True)
df_tot_0_r.columns = ['LSTM', 'LSTMX', 'LSTMC']

df_tot_1_r = pd.read_csv('Results/resu_rf.csv', index_col=(0)).T.iloc[:, :3].dropna(axis=0).reset_index(drop=True)
df_tot_1_r.columns = ['RF', 'RFX', 'RFC']

# Calculate counts for ARIMA
categories_ar, counts_tot_ar, counts_tot_r_ar, counts_diff_ar = calculate_counts(df_tot_r)
categories_lstm, counts_tot_lstm, counts_tot_r_lstm, counts_diff_lstm = calculate_counts(df_tot_0_r)
categories_rf, counts_tot_rf, counts_tot_r_rf, counts_diff_rf = calculate_counts(df_tot_1_r)

cat_ar = np.array([counts_tot_ar,counts_tot_r_ar])
cat_lstm = np.array([counts_tot_lstm,counts_tot_r_lstm])
cat_rf = np.array([counts_tot_rf,counts_tot_r_rf])

# Plotting
bar_width = 0.2
index = [0.3,0.7]

fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=True)
arima_bars = axs[0].bar(index, cat_ar[:,0], bar_width, label='Negative', color='lightgray')
axs[0].bar(index, cat_ar[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_ar[:,0])
axs[0].bar(index, cat_ar[:,2], bar_width, label='Positive', color='#454545', bottom=cat_ar[:,0]+cat_ar[:,1])
axs[0].set_title('ARIMA',fontsize=50)

# Plotting for LSTM
lstm_bars =axs[1].bar(index, cat_lstm[:,0], bar_width, label='Negative', color='lightgray')
axs[1].bar(index, cat_lstm[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_lstm[:,0])
axs[1].bar(index, cat_lstm[:,2], bar_width, label='Positive', color='#454545', bottom=cat_lstm[:,0]+cat_lstm[:,1])
axs[1].set_title('LSTM',fontsize=50)

# Plotting for RF
rf_bars =axs[2].bar(index, cat_rf[:,0], bar_width, label='Negative', color='lightgray')
axs[2].bar(index, cat_rf[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_rf[:,0])
axs[2].bar(index, cat_rf[:,2], bar_width, label='Positive', color='#454545', bottom=cat_rf[:,0]+cat_rf[:,1])
axs[2].set_title('RF',fontsize=50)

axs[0].plot(np.array([0.4,0.6]),cat_ar[:,0]+cat_ar[:,1],  linestyle='dotted', color='#454545')
axs[0].plot(np.array([0.4,0.6]),cat_ar[:,0],  linestyle='dotted', color='darkgray')
axs[1].plot(np.array([0.4,0.6]),cat_lstm[:,0]+cat_lstm[:,1],  linestyle='dotted', color='#454545')
axs[1].plot(np.array([0.4,0.6]),cat_lstm[:,0],  linestyle='dotted', color='darkgray')
axs[2].plot(np.array([0.4,0.6]),cat_rf[:,0]+cat_rf[:,1],  linestyle='dotted', color='#454545')
axs[2].plot(np.array([0.4,0.6]),cat_rf[:,0],  linestyle='dotted', color='darkgray')
for ax in axs:
    ax.set_xticks(index)
    ax.set_xticklabels(['Dynamic', 'Compound'],fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlim(0,1)
    if ax == axs[0]:
        ax.spines['left'].set_visible(True)
        ax.set_ylabel('Percentage',fontsize=50)
        ax.legend(fontsize=40)
        ax.tick_params(axis='y', labelsize=40)

plt.tight_layout()
plt.show()




# =============================================================================
# Features
# =============================================================================

df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)

# Load the feature datasets
t_cara = pd.read_csv('Datasets/hctsa_features.csv',index_col=(0))
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
data_m = pd.read_csv('Datasets/hctsa_datamatrix.csv',header=None)

# Analyze the AR model
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:, -1]  # Select the last column as y
X = data_c.iloc[:, :-1]  # Select all other columns as X

# Select the corresponding y values
p_valu=[]
model_val=[]
ind=[]
for i in range(len(X.columns)):
    if X.iloc[:,i].nunique() != 1:
        try:
            X_sub = sm.add_constant(X.iloc[:,i])
            y_na = y.loc[X_sub.index]
            model = sm.OLS(y_na,X_sub).fit()
            model_val.append(model.params.loc[X.columns[i]])
            p_valu.append(model.pvalues.loc[X.columns[i]])
            ind.append(i)
        except:
            pass
    
ar_sig = pd.DataFrame(np.array([model_val,p_valu]).T,index=ind)

# Repeat the same steps for RF and LSTM models
# RF

df_tot= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)


# Analyze the AR model
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:, -1]  # Select the last column as y
X = data_c.iloc[:, :-1]  # Select all other columns as X

p_valu=[]
model_val=[]
ind=[]
for i in range(len(X.columns)):
    if X.iloc[:,i].nunique() != 1:
        try:
            X_sub = sm.add_constant(X.iloc[:,i])
            y_na = y.loc[X_sub.index]
            model = sm.OLS(y_na,X_sub).fit()
            model_val.append(model.params.loc[X.columns[i]])
            p_valu.append(model.pvalues.loc[X.columns[i]])
            ind.append(i)
        except:
            pass
    
rf_sig = pd.DataFrame(np.array([model_val,p_valu]).T,index=ind)

# LSTM 

df_tot= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)

# Analyze the AR model
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:,-1]  # Select the last column as y
X = data_c.iloc[:,:-1]  # Select all other columns as X

p_valu=[]
model_val=[]
ind=[]
for i in range(len(X.columns)):
    if X.iloc[:,i].nunique() != 1:
        try:
            X_sub = sm.add_constant(X.iloc[:,i])
            y_na = y.loc[X_sub.index]
            model = sm.OLS(y_na,X_sub).fit()
            model_val.append(model.params.loc[X.columns[i]])
            p_valu.append(model.pvalues.loc[X.columns[i]])
            ind.append(i)
        except:
            pass
    
nn_sig = pd.DataFrame(np.array([model_val,p_valu]).T,index=ind)


ar_sig = ar_sig[ar_sig.iloc[:,1]<0.05]
rf_sig = rf_sig[rf_sig.iloc[:,1]<0.05]
nn_sig = nn_sig[nn_sig.iloc[:,1]<0.05]

common_index = nn_sig.index.intersection(rf_sig.index).intersection(ar_sig.index)
merged_df = pd.concat([ar_sig.loc[common_index],rf_sig.loc[common_index],nn_sig.loc[common_index]],axis=1)
merged_df = merged_df[merged_df.iloc[:, [0, 2, 4]].apply(lambda x: all(x > 0) or all(x < 0), axis=1)]
merged_df=merged_df.iloc[[2,6,7,8,9,10,11],:]

t_cara_sig = t_cara.iloc[merged_df.index]
t_cara_sig.index= t_cara_sig.index-1
t_cara_sig = pd.concat([t_cara_sig,merged_df.iloc[:,[0,2,4]]],axis=1)



for i in [3735,3781]:# t_cara_sig.index:
    df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
    df_tot=df_tot.T
    df_tot=df_tot.reset_index(drop=True)
    data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
    data_c = data_c[data_c.iloc[:,7730].notna()]
    y = data_c.iloc[:,-1]  # Select the last column as y
    X = data_c.iloc[:,:-1]  # Select all other columns as X
    X_sub = sm.add_constant(X.iloc[:,i])
    #X_sub=X_sub[X.iloc[:,i]>0]
    y_na = y.loc[X_sub.index]
    model = sm.OLS(y_na,X_sub).fit()
    plt.scatter(X_sub[i], y_na, color='red')
    x_values = np.linspace(X_sub[i].min(), X_sub[i].max(), 100)
    if len(model.params)==2:
        y_values = model.params.iloc[0] + model.params.iloc[1] * x_values  # Assuming a simple linear regression, adjust as needed
    else:
        y_values = model.params.iloc[0] * x_values 
    plt.plot(x_values, y_values, label='ARIMA', color='red')
    #plt.hlines(0,0,0.175)
    plt.title(t_cara_sig.loc[i]['Name'])
    plt.show()
    
    df_tot= pd.read_csv('Results/resu_rf.csv',index_col=(0))
    df_tot=df_tot.T
    df_tot=df_tot.reset_index(drop=True)
    data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
    data_c = data_c[data_c.iloc[:,7730].notna()]
    y = data_c.iloc[:,-1]  # Select the last column as y
    X = data_c.iloc[:,:-1]  # Select all other columns as X
    X_sub = sm.add_constant(X.iloc[:,i])
    #X_sub=X_sub[X.iloc[:,i]>0]
    y_na = y.loc[X_sub.index]
    model = sm.OLS(y_na,X_sub).fit()
    plt.scatter(X_sub[i], y_na, color='blue')
    x_values = np.linspace(X_sub[i].min(), X_sub[i].max(), 100)
    if len(model.params)==2:
        y_values = model.params.iloc[0] + model.params.iloc[1] * x_values  # Assuming a simple linear regression, adjust as needed
    else:
        y_values = model.params.iloc[0] * x_values  # Assuming a simple linear regression, adjust as needed
    plt.plot(x_values, y_values, label='RF', color='blue')
    plt.title(t_cara_sig.loc[i]['Name'])
    #plt.hlines(0,0,0.175)
    plt.show()
    
    df_tot= pd.read_csv('Results/resu_nn.csv',index_col=(0))
    df_tot=df_tot.T
    df_tot=df_tot.reset_index(drop=True)
    data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
    data_c = data_c[data_c.iloc[:,7730].notna()]
    y = data_c.iloc[:,-1]  # Select the last column as y
    X = data_c.iloc[:,:-1]  # Select all other columns as X
    X_sub = sm.add_constant(X.iloc[:,i])
    #X_sub=X_sub[X.iloc[:,i]>0]
    y_na = y.loc[X_sub.index]
    model = sm.OLS(y_na,X_sub).fit()
    plt.scatter(X_sub[i], y_na, color='orange')
    x_values = np.linspace(X_sub[i].min(), X_sub[i].max(), 100)
    if len(model.params)==2:
        y_values = model.params.iloc[0] + model.params.iloc[1] * x_values  # Assuming a simple linear regression, adjust as needed
    else:
        y_values = model.params.iloc[0] * x_values # Assuming a simple linear regression, adjust as needed
    plt.plot(x_values, y_values, label='LSTM', color='orange')
    plt.title(t_cara_sig.loc[i]['Name'])
    plt.legend()
    #plt.hlines(0,0,0.175)
    plt.show()

df = pd.read_csv('Datasets/hctsa_timeseries-data.csv',names=range(10000))
df=df.iloc[:,:1000]
df=pd.DataFrame(df)

test=data_m.iloc[:,[3735,3781]] #3735,3781
df_tot= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot_1= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_2= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot_2=df_tot_2.T
df_tot_2=df_tot_2.reset_index(drop=True)
data_c = pd.concat([test,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]),np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]),np.log(df_tot_2.iloc[:,0]/df_tot_2.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,0]>0.025]
data_c['mini'] = data_c[[0,1,2]].min(axis=1)
data_c = data_c.sort_values(['mini'],ascending=False)

for i in data_c.index[:3]:
    plt.plot(df.iloc[i,250:320],marker='o')
    plt.title(f'Improv_NN = {round(data_c.loc[i,0],2)} -Improv_RF = {round(data_c.loc[i,1],2)} - Improv_AR = {round(data_c.loc[i,2],2)} - Param1 = {data_c.loc[i,3735]} - Param2 = {data_c.loc[i,3781]}')
    plt.show()
    
    




df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:,-1]  # Select the last column as y
X = data_c.iloc[:,:-1]  # Select all other columns as X
X_sub = sm.add_constant(X.iloc[:,[3735,3781]])
X_sub=X_sub[(X.iloc[:,3735]>0.025) & (X.iloc[:,3735]<0.1) & (X.iloc[:,3781]>0.025) & (X.iloc[:,3781]<0.1) ]
y_na = y.loc[X_sub.index]
plt.boxplot(y_na,positions=[0])
print(ttest_1samp(y_na, 0))
df_tot= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:,-1]  # Select the last column as y
X = data_c.iloc[:,:-1]  # Select all other columns as X
X_sub = sm.add_constant(X.iloc[:,[3735,3781]])
X_sub=X_sub[(X.iloc[:,3735]>0.025) & (X.iloc[:,3735]<0.1) & (X.iloc[:,3781]>0.025) & (X.iloc[:,3781]<0.1) ]
y_na = y.loc[X_sub.index]
plt.boxplot(y_na,positions=[1])
print(ttest_1samp(y_na, 0))
df_tot= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
data_c = pd.concat([data_m,np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]
y = data_c.iloc[:,-1]  # Select the last column as y
X = data_c.iloc[:,:-1]  # Select all other columns as X
X_sub = sm.add_constant(X.iloc[:,[3735,3781]])
X_sub=X_sub[(X.iloc[:,3735]>0.025) & (X.iloc[:,3735]<0.1) & (X.iloc[:,3781]>0.025) & (X.iloc[:,3781]<0.1) ]
y_na = y.loc[X_sub.index]
plt.boxplot(y_na,positions=[2])
print(ttest_1samp(y_na, 0))
plt.show()

