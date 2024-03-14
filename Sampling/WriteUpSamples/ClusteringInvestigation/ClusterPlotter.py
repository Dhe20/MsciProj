#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
# plt.rcParams['figure.facecolor'] = 'white'

#197 broke
def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

title = 'Clustering'

investigated_characteristic = "clustering"
investigated_values = [30]
max_numbers = ["2" for i in range(len(investigated_values))]

means = []
stds = []
#for column in df.columns:
post_avg = []
N = np.array(investigated_values)
meanss = []
stdss = []
pos = []
df_N = pd.DataFrame()

for i in range(len(investigated_values)):
    #print(i)
    filename = "SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    df.dropna(inplace=True, axis=1)
    means = []
    stds = []
    for column in df.columns:
        pdf_single = df[column]/df[column].sum() #* (df.index[1] - df.index[0])
        #pdf_single.dropna(inplace=True)
        vals = np.array(pdf_single.index)
        mean = sum(pdf_single*vals)
        # means or modes
        #mean = vals[np.argmax(pdf_single*vals)]
        if mean==0:
            continue
        means.append(mean)
        stds.append(np.sqrt(sum((pdf_single*pdf_single.index**2))-mean**2))
    df_N[str(investigated_values[i])] = df.mean(axis=1)
    df_N[str(investigated_values[i])] = df_N[str(investigated_values[i])] / df_N[str(investigated_values[i])].sum()
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)


#%%
fig = plt.figure(figsize = (24,20))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=4,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[1,0])
ax3 = fig.add_subplot(spec[2,0])
ax4 = fig.add_subplot(spec[3,0])
#ax5 = fig.add_subplot(spec[4,0])
ax1_2 = fig.add_subplot(spec[0,1])
ax2_2 = fig.add_subplot(spec[1,1])
ax3_2 = fig.add_subplot(spec[2,1])
ax4_2 = fig.add_subplot(spec[3,1])
#ax5_2 = fig.add_subplot(spec[4,1])
#ax = [ax1,ax2, ax3, ax4, ax5, ax1_2,ax2_2, ax3_2, ax4_2, ax5_2]
ax = [ax1,ax2, ax3, ax4, ax1_2,ax2_2, ax3_2, ax4_2]



for i in ax:
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)



plt.subplots_adjust(wspace=0, hspace=0)

b = 10
bias_0 = np.mean(meanss[0])-70
bias_1 = np.mean(meanss[1])-70
bias_2 = np.mean(meanss[2])-70
bias_3 = np.mean(meanss[3])-70
# bias_4 = np.mean(meanss[4])-70
bias_err_0 = np.std(meanss[0])
bias_err_1 = np.std(meanss[1])
bias_err_2 = np.std(meanss[2])
bias_err_3 = np.std(meanss[3])


bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])
bias_2, bias_err_2 = expected(meanss[2], stdss[2])
bias_3, bias_err_3 = expected(meanss[3], stdss[3])
#bias_4, bias_err_4 = expected(meanss[4], stdss[4])


bias_0 -= 70
bias_1 -= 70
bias_2 -= 70
bias_3 -= 70
#bias_4 -= 70

means_xmin = 60
means_xmax = 80

#Means:

ax1.hist(meanss[0], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[means_xmin,means_xmax])
ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[means_xmin,means_xmax])
ax3.hist(meanss[2], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[means_xmin,means_xmax])
ax4.hist(meanss[3], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[means_xmin,means_xmax])
#ax5.hist(meanss[4], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[means_xmin,means_xmax])

ax1.vlines(x=bias_0+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1))
ax3.vlines(x=bias_2+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_2, bias_err_2))
ax4.vlines(x=bias_3+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_3, bias_err_3))
#ax5.vlines(x=bias_3+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_4, bias_err_4))
#
ax1.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax2.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax3.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax4.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
#ax5.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)


# ax1.set_ylim(0,0.49)
ax1.set_xlim(means_xmin,means_xmax)
# ax2.set_ylim(0,0.4)
ax2.set_xlim(means_xmin,means_xmax)
# ax3.set_ylim(0,0.2)
ax3.set_xlim(means_xmin,means_xmax)
# ax4.set_ylim(0,0.2)
ax4.set_xlim(means_xmin,means_xmax)
#ax5.set_xlim(means_xmin,means_xmax)


# ax1.grid(axis='x', ls='dashed', alpha=0.5)
# ax2.grid(axis='x', ls='dashed', alpha=0.5)
ax1.legend(fontsize=16)
ax2.legend(fontsize=16)
ax3.legend(fontsize=16)
ax4.legend(fontsize=16)
#ax5.legend(fontsize=16)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
#ax4.set_xticklabels([])
ax1_2.set_xticklabels([])
ax2_2.set_xticklabels([])
ax3_2.set_xticklabels([])
#ax4_2.set_xticklabels([])
# ax2.set_yticks([0.0,0.15,0.30])
# ax3.set_yticks([0.0,0.08,0.16])
# ax4.set_yticks([0.0,0.08,0.16])

#Stds:

stds_xmin = 0
stds_xmax = 5


ax1_2.hist(stdss[0], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[stds_xmin,stds_xmax])
ax2_2.hist(stdss[1], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[stds_xmin,stds_xmax])
ax3_2.hist(stdss[2], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[stds_xmin,stds_xmax])
ax4_2.hist(stdss[3], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[stds_xmin,stds_xmax])
#ax5_2.hist(stdss[4], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1, range=[stds_xmin,stds_xmax])

# per = []
# for i in range(len(percentage)):
#     per.append(100*(np.mean(percentage[i])))
# %%
