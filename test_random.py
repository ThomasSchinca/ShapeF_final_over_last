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

# Load ARIMA result data, drop missing values, and calculate the mean of each column
df_tot= pd.read_csv('Results/resu_ar_f.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot = df_tot.dropna(axis=0)
df_tot.columns=['AR','ARX','ARC']
print(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]), 0))
# plt.hist(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]),bins=20)
# #plt.ylim(0,200)
# plt.vlines(0,0,700,color='r')
# plt.show()


df_tot_0= pd.read_csv('Results/resu_nn_f.csv',index_col=(0))
df_tot_0=df_tot_0.T
df_tot_0=df_tot_0.iloc[:,3:]
df_tot_0=df_tot_0.reset_index(drop=True)
df_tot_0 = df_tot_0.dropna(axis=0)
df_tot_0.columns=['LSTM','LSTMX','LSTMC']
means_nn = df_tot_0.mean(axis=0).sort_values(ascending=False)
print(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]), 0))

# Repeat the same steps for RF results
df_tot_1= pd.read_csv('Results/resu_rf_f.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1 = df_tot_1.iloc[:,2:]
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_1 = df_tot_1.dropna(axis=0)
df_tot_1.columns=['RF','RFX','RFC']
means_rf = df_tot_1.mean(axis=0).sort_values(ascending=False)
print(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]), 0))

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


# ttest_rel(df_tot_0.iloc[:,0],df_tot_0.iloc[:,2])

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
#ax2.set_ylim(-0.0004, 0.002)

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
ax1.set_title('Placebo')
plt.show()


# =============================================================================
# Plot Improv
# =============================================================================

plt.plot(df_tot.iloc[:,0]-df_tot.iloc[:,1])
plt.title('AR-ARX')
plt.show()

plt.plot(df_tot.iloc[:,0]-df_tot.iloc[:,2])
plt.title('AR-ARC')
plt.show()


plt.plot(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,1])
plt.title('RF-RFX')
plt.show()

plt.plot(df_tot_1.iloc[:,0]-df_tot_1.iloc[:,2])
plt.title('RF-RFC')
plt.show()

plt.plot(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,1])
plt.title('NN-NNX')
plt.show()

plt.plot(df_tot_0.iloc[:,0]-df_tot_0.iloc[:,2])
plt.title('NN-NNC')
plt.show()


# =============================================================================
# Density 
# =============================================================================

def calculate_counts(df, df_r):
    log_ratios_tot = np.log(df.iloc[:, 0] / df.iloc[:, 2])
    neg_count_tot = np.sum(log_ratios_tot < 0)
    zero_count_tot = np.sum(log_ratios_tot == 0)
    pos_count_tot = np.sum(log_ratios_tot > 0)

    log_ratios_tot_r = np.log(df_r.iloc[:, 0] / df_r.iloc[:, 2])
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

# Read the data
df_tot = pd.read_csv('Results/resu_ar_f.csv', index_col=(0)).T.dropna(axis=0).reset_index(drop=True)
df_tot.columns = ['AR', 'ARX', 'ARC']

df_tot_r = pd.read_csv('Results/resu_ar.csv', index_col=(0)).T.dropna(axis=0).reset_index(drop=True)
df_tot_r.columns = ['AR', 'ARX', 'ARC']

df_tot_0_r = pd.read_csv('Results/resu_nn.csv', index_col=(0)).T.dropna(axis=1, how='all').dropna(axis=0).reset_index(drop=True)
df_tot_0_r.columns = ['LSTM', 'LSTMX', 'LSTMC']

df_tot_1_r = pd.read_csv('Results/resu_rf.csv', index_col=(0)).T.iloc[:, :3].dropna(axis=0).reset_index(drop=True)
df_tot_1_r.columns = ['RF', 'RFX', 'RFC']

# Calculate counts for ARIMA
categories_ar, counts_tot_ar, counts_tot_r_ar, counts_diff_ar = calculate_counts(df_tot, df_tot_r)

# Calculate counts for LSTM
categories_lstm, counts_tot_lstm, counts_tot_r_lstm, counts_diff_lstm = calculate_counts(df_tot_0, df_tot_0_r)

# Calculate counts for RF
categories_rf, counts_tot_rf, counts_tot_r_rf, counts_diff_rf = calculate_counts(df_tot_1, df_tot_1_r)

# Plotting
bar_width = 0.2
index = np.arange(len(categories_ar))

fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=True)

def plot_subplot(ax, categories, counts_tot, counts_tot_r, counts_diff, title, legend=False):
    hatch_patterns = ['', '/', '\\', 'x']
    bars = [
        ax.bar(index, counts_tot, bar_width, label='Placebo', color='lightgray', hatch=hatch_patterns[0]),
        ax.bar(index + bar_width, counts_tot_r, bar_width, label='Real Model', color='darkgray', hatch=hatch_patterns[1]),
        ax.bar(index + 2 * bar_width, counts_diff, bar_width, label='Difference', color='black', hatch=hatch_patterns[2])
    ]
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Remove legend if not required
    if not legend:
        ax.legend().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylabel('')
    else:
        ax.legend(loc='upper left')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(categories)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

plot_subplot(axs[0], categories_ar, counts_tot_ar, counts_tot_r_ar, counts_diff_ar, 'ARIMA',legend=True)
plot_subplot(axs[2], categories_lstm, counts_tot_lstm, counts_tot_r_lstm, counts_diff_lstm, 'LSTM')
plot_subplot(axs[1], categories_rf, counts_tot_rf, counts_tot_r_rf, counts_diff_rf, 'RF')
plt.show()





# test = np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2])
# test = test[test>np.log(5)]

# df = pd.read_csv('Datasets/hctsa_timeseries-data.csv',names=range(10000))
# df=df.iloc[:,:1000]
# scaler = MinMaxScaler(feature_range=(0,1))
# df=df.T
# df = scaler.fit_transform(df) 
# df=df.T
# df=pd.DataFrame(df)

# from pmdarima.arima import auto_arima
# for i in test.index:
#     arima = auto_arima(df.iloc[i,:800])
#     #if i in [178,182]:
#     plt.plot(df.iloc[i,:30])
#     plt.plot(arima.fittedvalues()[:30],color='r')
#     # else:
#     #     plt.plot(df.iloc[i,:800])
#     #     plt.plot(arima.fittedvalues(),color='r')
#     plt.title(f'{arima.order} - {test[i]}')    
#     plt.show()




def calculate_counts(df, df_r):
    log_ratios_tot = np.log(df.iloc[:, 0] / df.iloc[:, 2])
    neg_count_tot = np.sum(log_ratios_tot < 0)
    zero_count_tot = np.sum(log_ratios_tot == 0)
    pos_count_tot = np.sum(log_ratios_tot > 0)

    log_ratios_tot_r = np.log(df_r.iloc[:, 0] / df_r.iloc[:, 2])
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

# Read the data
df_tot = pd.read_csv('Results/resu_ar_f.csv', index_col=(0)).T.dropna(axis=0).reset_index(drop=True)
df_tot.columns = ['AR', 'ARX', 'ARC']

df_tot_r = pd.read_csv('Results/resu_ar.csv', index_col=(0)).T.dropna(axis=0).reset_index(drop=True)
df_tot_r.columns = ['AR', 'ARX', 'ARC']

df_tot_0_r = pd.read_csv('Results/resu_nn.csv', index_col=(0)).T.dropna(axis=1, how='all').dropna(axis=0).reset_index(drop=True)
df_tot_0_r.columns = ['LSTM', 'LSTMX', 'LSTMC']

df_tot_1_r = pd.read_csv('Results/resu_rf.csv', index_col=(0)).T.iloc[:, :3].dropna(axis=0).reset_index(drop=True)
df_tot_1_r.columns = ['RF', 'RFX', 'RFC']

# Calculate counts for ARIMA
categories_ar, counts_tot_ar, counts_tot_r_ar, counts_diff_ar = calculate_counts(df_tot, df_tot_r)
categories_lstm, counts_tot_lstm, counts_tot_r_lstm, counts_diff_lstm = calculate_counts(df_tot_0, df_tot_0_r)
categories_rf, counts_tot_rf, counts_tot_r_rf, counts_diff_rf = calculate_counts(df_tot_1, df_tot_1_r)

cat_ar = np.array([counts_tot_ar,counts_tot_r_ar])
cat_lstm = np.array([counts_tot_lstm,counts_tot_r_lstm])
cat_rf = np.array([counts_tot_rf,counts_tot_r_rf])

# Plotting
bar_width = 0.2
index = [0.3,0.7]

fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=True)
axs[0].bar(index, cat_ar[:,0], bar_width, label='Negative', color='lightgray')
axs[0].bar(index, cat_ar[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_ar[:,0])
axs[0].bar(index, cat_ar[:,2], bar_width, label='Positive', color='#454545', bottom=cat_ar[:,0]+cat_ar[:,1])
axs[0].set_title('ARIMA',fontsize=50)

# Plotting for LSTM
axs[1].bar(index, cat_lstm[:,0], bar_width, label='Negative', color='lightgray')
axs[1].bar(index, cat_lstm[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_lstm[:,0])
axs[1].bar(index, cat_lstm[:,2], bar_width, label='Positive', color='#454545', bottom=cat_lstm[:,0]+cat_lstm[:,1])
axs[1].set_title('LSTM',fontsize=50)

# Plotting for RF
axs[2].bar(index, cat_rf[:,0], bar_width, label='Negative', color='lightgray')
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
    ax.set_xticklabels(['Placebo', 'Real Model'],fontsize=40)
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