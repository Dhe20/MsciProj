from matplotlib.pyplot import cm

#matplotlib.rcParams['font.family'] = 'Computer Modern Serif'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.optimize import curve_fit
mpl.rcParams.update(mpl.rcParamsDefault)

title = 'Posterior Asymptotic Normality & Constraining Power'

investigated_characteristic = 'single_event_data'
investigated_values = [True]
max_numbers = ["0"]

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
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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
# Single event posterior plot


i = 0
filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
df = pd.read_csv(filename, index_col = 0)
dfp = df.prod(axis=1)
dfp /= dfp.sum()*(dfp.index[1]-dfp.index[0])



fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ymax=0.4
#cmap = cm.get_cmap('winter')
color = iter(cm.winter(np.linspace(0, 1, len(df.columns))))
for i in range(len(df.columns)):
    c = next(color)
    ax.plot(df.index, df[str(i)], c=c, alpha=0.75)
    if i == int(len(df.columns)/1.01):
        mid_c = c
ax.plot([],[], c = mid_c, alpha=0.5, label='Single Event Posteriors')
ax.plot(dfp.index, dfp.values, c='k', lw=5, label='Full posterior')
ax.vlines(x=70, ymin=0, ymax=ymax, color='r', lw=3, ls='dashed', label=r'$H_0 = 70$ km s$^{-1}$ Mpc$^{-1}$')


ax.set_xlim(50,100)
ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 24, framealpha=.5, loc = 'upper right')
ax.set_ylabel(r'$P\,(\,H_0\, \mid, \hat{{d}}_{GW},  \hat{{d}}_{G} \,)$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'png' # e.g .png, .svg, etc.

image_name = 'Plots//HighRes//single_event.'+image_format
plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)


image_format = 'png' # e.g .png, .svg, etc.

image_name = 'Plots//LowRes//single_event.'+image_format
plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=200)


plt.show()