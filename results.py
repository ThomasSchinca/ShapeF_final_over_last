import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from cycler import cycler
import random
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler
from functions import get_dynamic_clusters
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_rel,ttest_1samp
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helevetica']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Plot parameters 
#plot_params = {"text.usetex":True,"font.family":"serif","font.size":5,"xtick.labelsize":5,"ytick.labelsize":5,"axes.labelsize":5,"figure.titlesize":20,"figure.figsize":(5,8),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
#plt.rcParams.update(plot_params)

# Load the dataset 
df = pd.read_csv('Datasets/hctsa_timeseries-data.csv',names=range(10000))

############################
### Load the predictions ###
############################

# ARIMA
df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot = df_tot.dropna(axis=0)
df_tot.columns=['AR','ARX','ARC']
means_arima = df_tot.mean(axis=0)

print("ARIMA")

# MSE
print(round(means_arima,5))
print(round(df_tot.iloc[:,0].std()/np.sqrt(len(df_tot)),5))
print(round(df_tot.iloc[:,1].std()/np.sqrt(len(df_tot)),5))
print(round(df_tot.iloc[:,2].std()/np.sqrt(len(df_tot)),5))
print(round(ttest_rel(df_tot.iloc[:,0],df_tot.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot.iloc[:,0],df_tot.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).mean(),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).mean(),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).std()/np.sqrt(len(df_tot)),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).std()/np.sqrt(len(df_tot)),5))
print(round(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]), 0)[1],5))

# RF
df_tot_1= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1 = df_tot_1.iloc[:,:3]
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_1 = df_tot_1.dropna(axis=0)
df_tot_1.columns=['RF','RFX','RFC']
means_rf = df_tot_1.mean(axis=0)

print("RF")

# MSE
print(round(means_rf,5))
print(round(df_tot_1.iloc[:,0].std()/np.sqrt(len(df_tot_1)),5))
print(round(df_tot_1.iloc[:,1].std()/np.sqrt(len(df_tot_1)),5))
print(round(df_tot_1.iloc[:,2].std()/np.sqrt(len(df_tot_1)),5))
print(round(ttest_rel(df_tot_1.iloc[:,0],df_tot_1.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot_1.iloc[:,0],df_tot_1.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).mean(),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).mean(),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).std()/np.sqrt(len(df_tot_1)),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).std()/np.sqrt(len(df_tot_1)),5))
print(round(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]), 0)[1],5))

# LSTM
df_tot_0= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot_0=df_tot_0.T
df_tot_0=df_tot_0.reset_index(drop=True)
df_tot_0 = df_tot_0.dropna(axis=1,how='all')
df_tot_0 = df_tot_0.dropna(axis=0)
df_tot_0.columns=['LSTM','LSTMX','LSTMC']
means_nn = df_tot_0.mean(axis=0)

print("LSTM")

# MSE
print(round(means_nn,5))
print(round(df_tot_0.iloc[:,0].std()/np.sqrt(len(df_tot_0)),5))
print(round(df_tot_0.iloc[:,1].std()/np.sqrt(len(df_tot_0)),5))
print(round(df_tot_0.iloc[:,2].std()/np.sqrt(len(df_tot_0)),5))
print(round(ttest_rel(df_tot_0.iloc[:,0],df_tot_0.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot_0.iloc[:,0],df_tot_0.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).mean(),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).mean(),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).std()/np.sqrt(len(df_tot_0)),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).std()/np.sqrt(len(df_tot_0)),5))
print(round(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]), 0)[1],5))

###########################
### Example Time Series ###
###########################

# Get types of time series 
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
ts_kind=[]
for i in cara_df['Keywords']:
    text = i.split(',')
    #if text[0]=='synthetic':
    #    if (text[1]=='map') or (text[1]=='dynsys'):  
    #        ts_kind.append(text[2])
    #    else:    
    #        ts_kind.append(text[1])
    #else : 
    ts_kind.append(text[0])
ts_kind=pd.Series(ts_kind)        

random.seed(42)
fig, axes = plt.subplots(figsize = (13,10),nrows=7,ncols=12)
plt.setp(axes, xticks=[], yticks=[])
for row, ax in zip(random.sample(range(1000),k=84), axes.ravel()):
    row=random.sample(range(1000),k=1)[0]
    ax.set_title(ts_kind.iloc[row],size=12)
    ts_tot = df.iloc[row,:200]
    ax.plot(ts_tot,linestyle="solid")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.savefig("Results/example_series.eps",dpi=300,bbox_inches="tight")

#################
### Dendogram ###
#################

# Prepare inout
df=df.iloc[:,:1000]
scaler=MinMaxScaler(feature_range=(0,1))
df=df.T
df=scaler.fit_transform(df)
df=df.T
df=pd.DataFrame(df)

# k=6 and win=4
fig, axs = plt.subplots(figsize=(20, 7))
h= get_dynamic_clusters(df.iloc[85,:],n_clu=6,number_s=4)
h['cluster_shape']=h['cluster_shape'].reshape((6,4))
Z = linkage(h['cluster_shape'])
k=dendrogram(Z,ax=axs)
ku= list(map(int,k['ivl']))
ku_col=k['leaves_color_list']
dendrogram(Z, orientation='top', ax=axs,color_threshold=0,above_threshold_color='black')
co=0
for i, centroid in enumerate(h['cluster_shape']):
    centroid_ax = axs.inset_axes([(ku.index(co)/6), -0.45, 1/6, 0.4])
    centroid_ax.axis('off')
    centroid_ax.set_ylim(0,1)
    for k in h['sequences'][h['seqences_clusters']==i]:
        centroid_ax.plot(k, color="lightgray",linewidth=1)
    co=co+1
    centroid_ax.plot(centroid,color="black",marker="o",linewidth=3)

axs.set_xticks([*range(6)],['']*6)
axs.set_axis_off()
axs.set_yticks([0],[''])
plt.savefig("Results/example_dendogram_I.eps",dpi=300,bbox_inches="tight")

# k=6 and win=9
fig, axs = plt.subplots(figsize=(20, 7))
h= get_dynamic_clusters(df.iloc[85,:],n_clu=6,number_s=9)
h['cluster_shape']=h['cluster_shape'].reshape((6,9))
Z = linkage(h['cluster_shape'])
k=dendrogram(Z,ax=axs)
ku= list(map(int,k['ivl']))
ku_col=k['leaves_color_list']
dendrogram(Z, orientation='top', ax=axs,color_threshold=0,above_threshold_color='black')
co=0
for i, centroid in enumerate(h['cluster_shape']):
    centroid_ax = axs.inset_axes([(ku.index(co)/6), -0.45, 1/6, 0.4])
    centroid_ax.axis('off')
    centroid_ax.set_ylim(0,1)
    for k in h['sequences'][h['seqences_clusters']==i]:
        centroid_ax.plot(k, color="lightgray",linewidth=1)
    centroid_ax.plot(centroid, color="black",marker="o",linewidth=3)
    co=co+1
axs.set_xticks([*range(6)],['']*6)
axs.set_axis_off()
axs.set_yticks([0],[''])
plt.savefig("Results/example_dendogram_II.eps",dpi=300,bbox_inches="tight")

# k=4 and win=5
fig, axs = plt.subplots(figsize=(20, 7))
h= get_dynamic_clusters(df.iloc[85,:],n_clu=4,number_s=5)
h['cluster_shape']=h['cluster_shape'].reshape((4,5))
Z = linkage(h['cluster_shape'])
k=dendrogram(Z,ax=axs)
ku= list(map(int,k['ivl']))
ku_col=k['leaves_color_list']
dendrogram(Z, orientation='top', ax=axs,color_threshold=0,above_threshold_color='black')
co=0
for i, centroid in enumerate(h['cluster_shape']):
    centroid_ax = axs.inset_axes([(ku.index(co)/4), -0.45, 1/4, 0.4])
    centroid_ax.axis('off')
    centroid_ax.set_ylim(0,1)
    for k in h['sequences'][h['seqences_clusters']==i]:
        centroid_ax.plot(k, color="lightgray",linewidth=1)
    centroid_ax.plot(centroid, color="black",marker="o",linewidth=3)
    co=co+1
axs.set_xticks([*range(4)],['']*4)
axs.set_axis_off()
axs.set_yticks([0],[''])
plt.savefig("Results/example_dendogram_III.eps",dpi=300,bbox_inches="tight")

# k=8 and win=5
fig, axs = plt.subplots(figsize=(20, 7))
h= get_dynamic_clusters(df.iloc[85,:],n_clu=8,number_s=5)
h['cluster_shape']=h['cluster_shape'].reshape((8,5))
Z = linkage(h['cluster_shape'])
k=dendrogram(Z,ax=axs)
ku= list(map(int,k['ivl']))
ku_col=k['leaves_color_list']
dendrogram(Z, orientation='top', ax=axs,color_threshold=0,above_threshold_color='black')
co=0
for i, centroid in enumerate(h['cluster_shape']):
    centroid_ax = axs.inset_axes([(ku.index(co)/8), -0.45, 1/8, 0.4])
    centroid_ax.axis('off')
    centroid_ax.set_ylim(0,1)
    for k in h['sequences'][h['seqences_clusters']==i]:
        centroid_ax.plot(k, color="lightgray",linewidth=1)
    centroid_ax.plot(centroid, color="black",marker="o",linewidth=3)
    co=co+1
axs.set_xticks([*range(8)],['']*8)
axs.set_axis_off()
axs.set_yticks([0],[''])
plt.savefig("Results/example_dendogram_IV.eps",dpi=400,bbox_inches="tight")

######################
### Results---Plot ###
######################

# Bootstrapping to get means
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
    
# Get improvements
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

mean2_data = pd.DataFrame({'mean2': mean2,'std2': std2})
mean_data = pd.DataFrame({'mean': mean,'std': std})

# Prepare plot
blue_color = '#404040'  
orange_color = '#A0A0A0'  
def mse_formatter(x, pos):
    return '{:.1e}'.format(x)
def improvement_formatter(x, pos):
    return '{:.1e}'.format(x)
marker_size = 150
linewidth = 3
fonts=20
plt.rc('font', size=25)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
ax1.yaxis.label.set_color(blue_color)
ax1.tick_params(axis='y', colors=blue_color)
ax1.yaxis.set_major_formatter(FuncFormatter(mse_formatter))
ax1.scatter(mean_data.index, mean_data['mean'], color=blue_color, marker='o', s=marker_size)
ax1.errorbar(mean_data.index, mean_data['mean'], yerr=mean_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
ax2 = ax1.twinx()
ax2.grid(False)
ax2.yaxis.label.set_color(orange_color)
ax2.tick_params(axis='y', colors=orange_color)
ax2.yaxis.set_major_formatter(FuncFormatter(improvement_formatter))
ax2.scatter(mean2_data.index, mean2_data['mean2'], color=orange_color, marker='o', s=marker_size)
ax2.errorbar(mean2_data.index, mean2_data['mean2'], yerr=mean2_data['std2'], fmt='none', color=orange_color, linewidth=linewidth)
ax2.hlines(0, -0.5, 10.5, linestyles='--', color=orange_color, linewidth=linewidth)
ax2.set_yticks([-0.4,-0.3,-0.2,-0.1,0,0.1,0.2],["-0.4","-0.3","-0.2","-0.1","0","0.1","0.2"],fontsize=fonts)
ax1.set_ylabel('Mean squared error (MSE)', color=blue_color,fontsize=fonts)
ax2.set_ylabel('MSE improvement', color=orange_color,fontsize=fonts)
ax1.set_xticks([*range(11)],['ARIMA','D-ARIMA','ARIMA-C','','RF','D-RF','RF-C','','LSTM','D-LSTM','LSTM-C'],fontsize=fonts,rotation=45)
ax1.set_yticks([0.009,0.01,0.011,0.012,0.013,0.014,0.015],["0.009","0.01","0.011","0.012","0.013","0.014","0.015"],fontsize=fonts)
ax1.grid(False)
plt.savefig("Results/mse_improv.eps",dpi=300,bbox_inches="tight")
plt.show()

#####################
### Scatter plots ###
#####################

# Difference MSE ARIMA and D-ARIMA
results={"country":[],"change":[]}
for i in range(len(df_tot)): 
    results["country"].append(i)
    results["change"].append(df_tot.iloc[i,0]-df_tot.iloc[i,1])   
results=pd.DataFrame(results)
results.sort_values(by=['change'], ascending=True, inplace=True)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
y=list(range(0,840))
plt.scatter(y,results.sort_values(by=['change'], ascending=True).change, s=10, alpha=1, color='black', marker="o")
plt.xticks([])
plt.axhline(y=0, c="gray", linewidth=1,linestyle="dashed") 
plt.title('Difference in MSE ARIMA vs. D-ARIMA',size=25)
plt.yticks([-0.02,0,0.02,0.04,0.06,0.08,0.1,0.12,0.14],["-0.02","0","0.02","0.04","0.06","0.08","0.1","0.12","0.14"],size=20)
results=results.reset_index()
last_10 = results.tail(10)
for i in range(len(last_10)):
    plt.text(last_10.index[i],last_10['change'].iloc[i],f"{last_10['country'].iloc[i]}",
             fontsize=15, ha='left', color='black')
plt.savefig("Results/scatter1a.eps",dpi=300,bbox_inches="tight")

# Difference MSE RF and D-RF
results={"country":[],"change":[]}
for i in range(len(df_tot_1)): 
    results["country"].append(i)
    results["change"].append(df_tot_1.iloc[i,0]-df_tot_1.iloc[i,1])   
results=pd.DataFrame(results)
results.sort_values(by=['change'], ascending=True, inplace=True)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
y=list(range(0,840))
plt.scatter(y,results.sort_values(by=['change'], ascending=True).change, s=10, alpha=1, color='black', marker="o")
plt.xticks([])
plt.axhline(y=0, c="gray", linewidth=1,linestyle="dashed") 
plt.title('Difference in MSE RF vs. D-RF',size=25)
plt.yticks([-0.02,-0.01,0,0.01,0.02],["-0.02","-0.01","0","0.01","0.02"],size=20)
results=results.reset_index()
last_10 = pd.concat([results.tail(5),results.head(5)])
for i in range(len(last_10)):
    plt.text(last_10.index[i],last_10['change'].iloc[i],f"{last_10['country'].iloc[i]}",
             fontsize=15, ha='left', color='black')
plt.savefig("Results/scatter2a.eps",dpi=300,bbox_inches="tight")

# Difference MSE LSTM and D-LSTM
results={"country":[],"change":[]}
for i in range(len(df_tot_0)): 
    results["country"].append(i)
    results["change"].append(df_tot_0.iloc[i,0]-df_tot_0.iloc[i,1])   
results=pd.DataFrame(results)
results.sort_values(by=['change'], ascending=True, inplace=True)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
y=list(range(0,840))
plt.scatter(y,results.sort_values(by=['change'], ascending=True).change, s=10, alpha=1, color='black', marker="o")
plt.xticks([])
plt.axhline(y=0, c="gray", linewidth=1,linestyle="dashed") 
plt.title('Difference in MSE LSTM vs. D-LSTM',size=25)
plt.yticks([-0.06,-0.04,-0.02,0,0.02],["-0.06","-0.04","-0.02","0","0.02"],size=20)
results=results.reset_index()
last_10 = pd.concat([results.tail(5),results.head(5)])
for i in range(len(last_10)):
    plt.text(last_10.index[i],last_10['change'].iloc[i],f"{last_10['country'].iloc[i]}",
             fontsize=15, ha='left', color='black')
plt.savefig("Results/scatter3a.eps",dpi=300,bbox_inches="tight")

#################################
### Results---Random clusters ###
#################################

# ARIMA
df_tot= pd.read_csv('Results/resu_ar_f.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot = df_tot.dropna(axis=0)
df_tot.columns=['AR','ARX','ARC']
means_arima = df_tot.mean(axis=0)

print("ARIMA")

# MSE
print(round(means_arima,5))
print(round(df_tot.iloc[:,0].std()/np.sqrt(len(df_tot)),5))
print(round(df_tot.iloc[:,1].std()/np.sqrt(len(df_tot)),5))
print(round(df_tot.iloc[:,2].std()/np.sqrt(len(df_tot)),5))
print(round(ttest_rel(df_tot.iloc[:,0],df_tot.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot.iloc[:,0],df_tot.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).mean(),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).mean(),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]).std()/np.sqrt(len(df_tot)),5))
print(round(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]).std()/np.sqrt(len(df_tot)),5))
print(round(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,2]), 0)[1],5))

# RF
df_tot_1= pd.read_csv('Results/resu_rf_f.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1 = df_tot_1.iloc[:,2:]
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_1 = df_tot_1.dropna(axis=0)
df_tot_1.columns=['RF','RFX','RFC']
means_rf = df_tot_1.mean(axis=0)

print("RF")

# MSE
print(round(means_rf,5))
print(round(df_tot_1.iloc[:,0].std()/np.sqrt(len(df_tot_1)),5))
print(round(df_tot_1.iloc[:,1].std()/np.sqrt(len(df_tot_1)),5))
print(round(df_tot_1.iloc[:,2].std()/np.sqrt(len(df_tot_1)),5))
print(round(ttest_rel(df_tot_1.iloc[:,0],df_tot_1.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot_1.iloc[:,0],df_tot_1.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).mean(),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).mean(),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]).std()/np.sqrt(len(df_tot_1)),5))
print(round(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]).std()/np.sqrt(len(df_tot_1)),5))
print(round(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,2]), 0)[1],5))

# LSTM
df_tot_0= pd.read_csv('Results/resu_nn_f.csv',index_col=(0))
df_tot_0=df_tot_0.T
df_tot_0=df_tot_0.iloc[:,3:]
df_tot_0=df_tot_0.reset_index(drop=True)
df_tot_0 = df_tot_0.dropna(axis=0)
df_tot_0.columns=['LSTM','LSTMX','LSTMC']
means_nn = df_tot_0.mean(axis=0)

print("LSTM")

# MSE
print(round(means_nn,5))
print(round(df_tot_0.iloc[:,0].std()/np.sqrt(len(df_tot_0)),5))
print(round(df_tot_0.iloc[:,1].std()/np.sqrt(len(df_tot_0)),5))
print(round(df_tot_0.iloc[:,2].std()/np.sqrt(len(df_tot_0)),5))
print(round(ttest_rel(df_tot_0.iloc[:,0],df_tot_0.iloc[:,1])[1],5))
print(round(ttest_rel(df_tot_0.iloc[:,0],df_tot_0.iloc[:,2])[1],5))

# MSE improvement
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).mean(),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).mean(),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]).std()/np.sqrt(len(df_tot_0)),5))
print(round(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]).std()/np.sqrt(len(df_tot_0)),5))
print(round(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]), 0)[1],5))
print(round(ttest_1samp(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,2]), 0)[1],5))

########################################
### Results---Plot---Random clusters ###
########################################

# Bootstrapping to get means
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
    
# Get improvements
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

mean2_data = pd.DataFrame({'mean2': mean2,'std2': std2})
mean_data = pd.DataFrame({'mean': mean,'std': std})

# Prepare plot
blue_color = '#404040'  
orange_color = '#A0A0A0'  
def mse_formatter(x, pos):
    return '{:.1e}'.format(x)
def improvement_formatter(x, pos):
    return '{:.1e}'.format(x)
marker_size = 150
linewidth = 3
fonts=20
plt.rc('font', size=25)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
ax1.yaxis.label.set_color(blue_color)
ax1.tick_params(axis='y', colors=blue_color)
ax1.yaxis.set_major_formatter(FuncFormatter(mse_formatter))
ax1.scatter(mean_data.index, mean_data['mean'], color=blue_color, marker='o', s=marker_size)
ax1.errorbar(mean_data.index, mean_data['mean'], yerr=mean_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
ax2 = ax1.twinx()
ax2.grid(False)
ax2.yaxis.label.set_color(orange_color)
ax2.tick_params(axis='y', colors=orange_color)
ax2.yaxis.set_major_formatter(FuncFormatter(improvement_formatter))
ax2.scatter(mean2_data.index, mean2_data['mean2'], color=orange_color, marker='o', s=marker_size)
ax2.errorbar(mean2_data.index, mean2_data['mean2'], yerr=mean2_data['std2'], fmt='none', color=orange_color, linewidth=linewidth)
ax2.hlines(0, -0.5, 10.5, linestyles='--', color=orange_color, linewidth=linewidth)
ax2.set_yticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1],["-0.5","-0.4","-0.3","-0.2","-0.1","0","0.1"],fontsize=fonts)
ax1.set_ylabel('Mean squared error (MSE)', color=blue_color,fontsize=fonts)
ax2.set_ylabel('MSE improvement', color=orange_color,fontsize=fonts)
ax1.set_xticks([*range(11)],['ARIMA','D-ARIMA','ARIMA-C','','RF','D-RF','RF-C','','LSTM','D-LSTM','LSTM-C'],fontsize=fonts,rotation=45)
ax1.set_yticks([0.009,0.01,0.011,0.012,0.013,0.014],["0.009","0.01","0.011","0.012","0.013","0.014"],fontsize=fonts)
ax1.grid(False)
plt.savefig("Results/mse_improv_random.eps",dpi=300,bbox_inches="tight")
plt.show()

####################
### Scatter plot ###
####################

# Difference MSE ARIMA and ARIMA-C for random clusters
results={"country":[],"change":[],"improve":[]}
for i in range(len(df_tot)): 
    results["country"].append(i)
    results["change"].append(df_tot.iloc[i,0]-df_tot.iloc[i,2])
    results["improve"].append(np.log(df_tot.iloc[i,0]/df_tot.iloc[i,2]))  
results=pd.DataFrame(results)
results.sort_values(by=['change'], ascending=True, inplace=True)

# Plot
fig, ax1 = plt.subplots(figsize=(12,8))
y=list(range(0,835))
ax1.scatter(y,results.sort_values(by=['change'], ascending=True).change, s=10, alpha=1, color='black', marker="o")
ax2 = ax1.twinx()
ax2.scatter(y,results.sort_values(by=['improve'], ascending=True).improve, s=10, alpha=1, color='gray', marker="o")
plt.xticks([])
plt.axhline(y=0, c="gray", linewidth=1,linestyle="dashed") 
plt.title('Difference in MSE ARIMA vs. ARIMA-C',size=25)
ax1.set_yticks([0,0.01,0.02,0.03,0.04,0.05],["0","0.01","0.02","0.03","0.04","0.05"],size=20)
ax2.set_yticks([-2,0,2,4,6,8],["-2","0","2","4","6","8"],size=20)
patch1 = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Difference')
patch2 = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Log-ratio')
plt.legend(handles=[patch1, patch2],fontsize='15')         
plt.savefig("Results/scatter4a.eps",dpi=300,bbox_inches="tight")

############################################################
### Distributions of positive, zero, negative log ratios ###
############################################################

# Define function
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
print(np.round(counts_tot_r_ar,2))

# Calculate counts for RF
categories_rf, counts_tot_rf, counts_tot_r_rf, counts_diff_rf = calculate_counts(df_tot_1, df_tot_1_r)
print(np.round(counts_tot_r_rf,2))

# Calculate counts for LSTM
categories_lstm, counts_tot_lstm, counts_tot_r_lstm, counts_diff_lstm = calculate_counts(df_tot_0, df_tot_0_r)
print(np.round(counts_tot_r_lstm,2))

##################################################
### Distributions for random and real clusters ###
##################################################

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

cat_ar = np.array([counts_tot_ar,counts_tot_r_ar])
cat_lstm = np.array([counts_tot_lstm,counts_tot_r_lstm])
cat_rf = np.array([counts_tot_rf,counts_tot_r_rf])

# Plotting
bar_width = 0.2
index = [0.3,0.7]

fig, axs = plt.subplots(1, 3, figsize=(30, 12), sharey=True)
axs[0].bar(index, cat_ar[:,0], bar_width, label='Negative', color='gainsboro')
axs[0].text(0.255, 12, round(cat_ar[:,0][0],2), fontsize=30, color='black')
axs[0].text(0.655, 10, round(cat_ar[:,0][1],2), fontsize=30, color='black')
axs[0].bar(index, cat_ar[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_ar[:,0])
axs[0].text(0.255, 51, round(cat_ar[:,1][0],2), fontsize=30, color='black')
axs[0].text(0.655, 43, round(cat_ar[:,1][1],2), fontsize=30, color='black')
axs[0].bar(index, cat_ar[:,2], bar_width, label='Positive', color='dimgray', bottom=cat_ar[:,0]+cat_ar[:,1])
axs[0].text(0.255, 87, round(cat_ar[:,2][0],2), fontsize=30, color='black')
axs[0].text(0.655, 79, round(cat_ar[:,2][1],2), fontsize=30, color='black')
axs[0].set_title('ARIMA',fontsize=50)

# Plotting for RF
axs[1].bar(index, cat_rf[:,0], bar_width, label='Negative', color='gainsboro')
axs[1].text(0.255, 9, round(cat_rf[:,0][0],2), fontsize=30, color='black')
axs[1].text(0.655, 10, round(cat_rf[:,0][1],2), fontsize=30, color='black')
axs[1].bar(index, cat_rf[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_rf[:,0])
axs[1].text(0.255, 47, round(cat_rf[:,1][0],2), fontsize=30, color='black')
axs[1].text(0.655, 39, round(cat_rf[:,1][1],2), fontsize=30, color='black')
axs[1].bar(index, cat_rf[:,2], bar_width, label='Positive', color='dimgray', bottom=cat_rf[:,0]+cat_rf[:,1])
axs[1].text(0.255, 85, round(cat_rf[:,2][0],2), fontsize=30, color='black')
axs[1].text(0.655, 78, round(cat_rf[:,2][1],2), fontsize=30, color='black')
axs[1].set_title('RF',fontsize=50)

# Plotting for LSTM
axs[2].bar(index, cat_lstm[:,0], bar_width, label='Negative', color='gainsboro')
axs[2].text(0.255, 6, round(cat_lstm[:,0][0],2), fontsize=30, color='black')
axs[2].text(0.655, 7, round(cat_lstm[:,0][1],2), fontsize=30, color='black')
axs[2].bar(index, cat_lstm[:,1], bar_width, label='Zero', color='darkgray', bottom=cat_lstm[:,0])
axs[2].text(0.255, 51, round(cat_lstm[:,1][0],2), fontsize=30, color='black')
axs[2].text(0.655, 43, round(cat_lstm[:,1][1],2), fontsize=30, color='black')
axs[2].bar(index, cat_lstm[:,2], bar_width, label='Positive', color='dimgray', bottom=cat_lstm[:,0]+cat_lstm[:,1])
axs[2].text(0.255, 92, round(cat_lstm[:,2][0],2), fontsize=30, color='black')
axs[2].text(0.655, 84, round(cat_lstm[:,2][1],2), fontsize=30, color='black')
axs[2].set_title('LSTM',fontsize=50)

axs[0].plot(np.array([0.4,0.6]),cat_ar[:,0]+cat_ar[:,1],  linestyle='dotted', color='#454545',linewidth=3)
axs[0].plot(np.array([0.4,0.6]),cat_ar[:,0]+cat_ar[:,1]+cat_ar[:,2],  linestyle='dotted', color='#454545',linewidth=3)
axs[1].plot(np.array([0.4,0.6]),cat_rf[:,0]+cat_rf[:,1],  linestyle='dotted', color='#454545',linewidth=3)
axs[1].plot(np.array([0.4,0.6]),cat_rf[:,0]+cat_rf[:,1]+cat_rf[:,2],  linestyle='dotted', color='#454545',linewidth=3)
axs[2].plot(np.array([0.4,0.6]),cat_lstm[:,0]+cat_lstm[:,1],  linestyle='dotted', color='#454545',linewidth=3)
axs[2].plot(np.array([0.4,0.6]),cat_lstm[:,0]+cat_lstm[:,1]+cat_lstm[:,2],  linestyle='dotted', color='#454545',linewidth=3)

for ax in axs:
    ax.set_xticks(index)
    ax.set_xticklabels(['Random', 'Real'],fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlim(0,1)
    if ax == axs[0]:
        ax.spines['left'].set_visible(True)
        ax.set_ylabel('Percentage',fontsize=50)
        ax.tick_params(axis='y', labelsize=40)
patch3 = mpatches.Patch(color='dimgray', label='Positive')   
patch2 = mpatches.Patch(color='darkgray', label='Zero')        
patch1 = mpatches.Patch(color='gainsboro', label='Negative')        
fig.legend(handles=[patch1,patch2,patch3],fontsize=40,ncol=3,loc='center', bbox_to_anchor=(0.5,-0.05))
plt.tight_layout()
plt.savefig("Results/bar_plot_distributions.eps",dpi=300,bbox_inches="tight")
plt.show()

##################
### Conditions ###
##################

# ARIMA #
df_tot= pd.read_csv('Results/resu_ar.csv',index_col=(0))
df_tot=df_tot.T
df_tot=df_tot.reset_index(drop=True)
df_tot = df_tot.dropna(axis=0)
df_tot.columns=['AR','ARX','ARC']

# Load the feature datasets
t_cara = pd.read_csv('Datasets/hctsa_features.csv',index_col=(0))
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
data_m = pd.read_csv('Datasets/hctsa_datamatrix.csv',header=None)

# Get types of time series 
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
names=pd.Series(ts_kind)       
names.name="ts_type"
data_c = pd.concat([data_m, names],axis=1)

names=pd.Series(np.log(df_tot.iloc[:,0]/df_tot.iloc[:,1]))       
names.name="log_ratio"
data_c = pd.concat([data_c,names],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]

grouped_mean = data_c.groupby('ts_type')['log_ratio'].mean()
grouped_mean= grouped_mean.reset_index()
grouped_mean.sort_values(by=['log_ratio'], ascending=True, inplace=True)
grouped_mean=grouped_mean.dropna()

# Plot best performing time series
def remove_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# Plot best performing time series    
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(reversed(list(grouped_mean.tail(7).ts_type))):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0                       
                
plt.tight_layout()
plt.savefig("Results/cases_best_grid_arima.eps",dpi=300,bbox_inches="tight")
plt.show()

# Plot worst performing time series
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(grouped_mean.head(7).ts_type):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0            
               
plt.tight_layout()
plt.savefig("Results/cases_worst_grid_arima.eps",dpi=300,bbox_inches="tight")
plt.show()

# RF #
df_tot_1= pd.read_csv('Results/resu_rf.csv',index_col=(0))
df_tot_1=df_tot_1.T
df_tot_1 = df_tot_1.iloc[:,:3]
df_tot_1=df_tot_1.reset_index(drop=True)
df_tot_1 = df_tot_1.dropna(axis=0)
df_tot_1.columns=['RF','RFX','RFC'] 

# Load the feature datasets
t_cara = pd.read_csv('Datasets/hctsa_features.csv',index_col=(0))
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
data_m = pd.read_csv('Datasets/hctsa_datamatrix.csv',header=None)

# Get types of time series 
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
names=pd.Series(ts_kind)       
names.name="ts_type"
data_c = pd.concat([data_m, names],axis=1)

names=pd.Series(np.log(df_tot_1.iloc[:,0]/df_tot_1.iloc[:,1]))       
names.name="log_ratio"
data_c = pd.concat([data_c,names],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]

grouped_mean = data_c.groupby('ts_type')['log_ratio'].mean()
grouped_mean= grouped_mean.reset_index()
grouped_mean.sort_values(by=['log_ratio'], ascending=True, inplace=True)
grouped_mean=grouped_mean.dropna()

# Plot best performing time series
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(reversed(list(grouped_mean.tail(7).ts_type))):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0                  
                
plt.tight_layout()
plt.savefig("Results/cases_best_grid_rf.eps",dpi=300,bbox_inches="tight")
plt.show()

# Plot worst performing time series
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(grouped_mean.head(7).ts_type):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0            
        
plt.tight_layout()
plt.savefig("Results/cases_worst_grid_rf.eps",dpi=300,bbox_inches="tight")
plt.show()

# LSTM #
df_tot_0= pd.read_csv('Results/resu_nn.csv',index_col=(0))
df_tot_0=df_tot_0.T
df_tot_0=df_tot_0.reset_index(drop=True)
df_tot_0 = df_tot_0.dropna(axis=1,how='all')
df_tot_0 = df_tot_0.dropna(axis=0)
df_tot_0.columns=['LSTM','LSTMX','LSTMC']    

# Load the feature datasets
t_cara = pd.read_csv('Datasets/hctsa_features.csv',index_col=(0))
cara_df = pd.read_csv('Datasets/hctsa_timeseries-info.csv',index_col=0)
data_m = pd.read_csv('Datasets/hctsa_datamatrix.csv',header=None)
   
# Get types of time series 
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
names=pd.Series(ts_kind)       
names.name="ts_type"
data_c = pd.concat([data_m, names],axis=1)
frequency = data_c['ts_type'].value_counts()

names=pd.Series(np.log(df_tot_0.iloc[:,0]/df_tot_0.iloc[:,1]))       
names.name="log_ratio"
data_c = pd.concat([data_c,names],axis=1)
data_c = data_c[data_c.iloc[:,7730].notna()]

grouped_mean = data_c.groupby('ts_type')['log_ratio'].mean()
grouped_mean= grouped_mean.reset_index()
grouped_mean.sort_values(by=['log_ratio'], ascending=True, inplace=True)
grouped_mean=grouped_mean.dropna()

# Plot best performing time series  
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(reversed(list(grouped_mean.tail(7).ts_type))):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0            
        
plt.tight_layout()
plt.savefig("Results/cases_best_grid_lstm.eps",dpi=300,bbox_inches="tight")
plt.show()


# Plot worst performing time series
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
flag=True
while col<5:
    for t in list(grouped_mean.head(7).ts_type):
        if len(data_c.loc[data_c["ts_type"]==t].index)>=5:
            for p in range(5):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0
        elif len(data_c.loc[data_c["ts_type"]==t].index)<5:

            for p in range(len(data_c.loc[data_c["ts_type"]==t].index)):
                if col==5:
                    break
                axs[col, row].plot(df.loc[data_c.loc[data_c["ts_type"]==t].index].iloc[p], linewidth=2, color='black')
                axs[col, row].set_title(f"{t}",size=30)
                remove_box(axs[col, row])
                axs[col, row].set_xticks([])
                axs[col, row].set_yticks([])
                row=row+1
                if row==5:
                    col=col+1
                    row=0            
        
plt.tight_layout()
plt.savefig("Results/cases_worst_grid_lstm.eps",dpi=300,bbox_inches="tight")
plt.show()













