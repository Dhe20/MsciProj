#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sps
from scipy.optimize import curve_fit
import seaborn as sn
from matplotlib import gridspec, collections
from Sampling.ClassSamples import Sampler
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True
plt.style.use('default')

f = 4*np.pi/375
c = 1.5*32*np.pi/3000


investigated_values = np.array([1.0]) #,0.5])
investigated_characteristic = 'bvmf_proportional_p_det_many'
investigated_values = [False, True]
max_numbers = ['0']*2

def bias_dist(x, H=70):
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps

def C_I_samp(x):
    H = np.random.uniform(50,100)
    H = 70
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps



def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

fig = plt.figure(figsize = (12,8))
# create grid for different subplots
# spec = gridspec.GridSpec(ncols=1, nrows=2,
#                         wspace=0.2,
#                          hspace=0.3)

ax1 = fig.add_subplot()

meanss = []
stdss = []
pos = []

p_i_s = []
c_i_s = []

for i in range(len(investigated_values)):
    #print(i)
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    means = []
    stds = []
    inc = df.index[1]-df.index[0]
    p_i_s.append(bias_dist(df))
    c_i_s.append(C_I_samp(df))
    for column in df.columns:
        if int(column)>999:
            continue
        if is_unique(df[column]):
            print('Gotcha')
            continue
        pdf_single = df[column]/(inc * df[column].sum())
        #print(df[column].sum())
        pdf_single.dropna(inplace=True)
        vals = np.array(pdf_single.index)
        mean = sum(inc * pdf_single*vals)
        # means or modes
        #mean = vals[np.argmax(pdf_single*vals)]
        if mean==0:
            continue
        means.append(mean)
        stds.append(np.sqrt(sum((inc*pdf_single*pdf_single.index**2))-mean**2))
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
B = 15
bins = np.linspace(0,1,B+1)
NN = len(p_i_s[0])
u_p = 0.995
d_p = 0.005

c = ['b', 'r']
vals = ['Not Included', 'Included']
for i in range(1,2):
    ax.hist(p_i_s[i], bins=bins, density=0, linewidth=5, histtype='step',edgecolor= 'dodgerblue', lw=4, label=vals[i])

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

upper_band = sps.binom.ppf(u_p, NN, 1/B)
lower_band = sps.binom.ppf(d_p, NN, 1/B)

ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='k', alpha=0.2, label=r'${:.0f}\%$ confidence band'.format(100*(u_p-d_p)))


u_p = 0.95
d_p = 0.05

upper_band_2 = sps.binom.ppf(u_p, NN, 1/B)
lower_band_2 = sps.binom.ppf(d_p, NN, 1/B)

ax.tick_params(axis='both', which='major', direction='in', labelsize=20, size=8, width=3, pad = 12)

ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='k', ls='dashed', lw=2)
ax.fill_between(bins, [lower_band_2]*(B+1), [upper_band_2]*(B+1), color='k', alpha=0.4, label=r'${:.0f}\%$ confidence band'.format(100*(u_p-d_p)))
#ax.set_title('{} Average Events'.format(selection_accounted[j]))
ax.set_xlim(-0.003,1.003)
#ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.set_xlabel('Rank Statistic', fontsize=24, labelpad=15)
ax.set_ylabel('Number of Samples', fontsize=24, labelpad=15)

plt.legend(fontsize=20)

image_format = 'png' # e.g .png, .svg, etc.

image_name = 'Plots//HighRes//RankStatPlot.'+image_format
plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)


image_format = 'png' # e.g .png, .svg, etc.

image_name = 'Plots//LowRes//RankStatPlot.'+image_format
plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=200)


plt.show()

plt.show()