#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections

from matplotlib.pyplot import cm
import matplotlib
from matplotlib.ticker import MultipleLocator

#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True


#%%
#SampleUniverse_3_50_0.1_50_3508.csv
df = pd.read_csv("PosteriorData\\SampleUniverse_trial_survey_completeness_2.5464790894703255e-07_0.csv", index_col = 0)
df = pd.read_csv("PosteriorData\\SampleUniverse_redshift_uncertainty_corrected_approx_0.005_0.csv", index_col = 0)
df = pd.read_csv("PosteriorData\\SampleUniverse_redshift_uncertainty_corrected_approx_0.005_0.csv", index_col = 0)
df = pd.read_csv("PosteriorData\\SampleUniverse_delta_D_5_average_events_3.409_0.csv", index_col = 0)

df = pd.read_csv('PosteriorData/SampleUniverse_bvmf_proportional_p_det_many_True_0.csv', index_col=0)

#df = pd.read_csv('PosteriorData/SampleUniverse_selection_effects_standard_3_True_0.csv', index_col=0)

df = pd.read_csv('PosteriorData/SampleUniverse_sel_eff_standard_01_True_0.csv', index_col=0)

df = pd.read_csv('PosteriorData/SampleUniverse_'+b)
df.dropna(inplace=True, axis=1)

spec = gridspec.GridSpec(ncols=1, nrows=3,
                         height_ratios=[4, 1, 1], wspace=0.2,
                         hspace=0.2)

fig = plt.figure(figsize = (32,12))


# create grid for different subplots
spec = gridspec.GridSpec(ncols=3, nrows=1,
                         wspace=0.38,
                         hspace=0.2)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])

means = []
stds = []

for column in df.columns:
    pdf_single = df[column]/df[column].sum()
    #ax1.plot(pdf_single)
    mean = sum(pdf_single*df.index)
    means.append(mean)
    stds.append(np.sqrt(sum((pdf_single*df.index**2))-mean**2))

N = 30
indices = np.argsort(means[:N])
cmap = cm.get_cmap('winter')
color = iter(cm.winter_r(np.linspace(0, 1, N)))

for i in indices:
    c = next(color)
    ax1.plot(df[str(i)], c=c, lw=4)

bias_0, bias_err_0 = expected(means, stds)
std_u = np.mean(stds)
std_std = np.std(stds)

ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.02))

ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(MultipleLocator(0.02))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))

ax3.xaxis.set_major_locator(MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(MultipleLocator(0.02))
ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
ax3.yaxis.set_major_locator(MultipleLocator(0.5))

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2.5)
    ax2.spines[axis].set_linewidth(2.5)
    ax3.spines[axis].set_linewidth(2.5)

ax1.tick_params(axis='both', top=True, right=True, which='major', direction='in', labelsize=35, size=9, width=3, pad = 15)
ax2.tick_params(axis='both', top=True, right=True, which='major', direction='in', labelsize=35, size=9, width=3, pad = 15)
ax3.tick_params(axis='both', top=True, right=True, which='major', direction='in', labelsize=35, size=9, width=3, pad = 15)

ax1.tick_params(axis='both', which='minor', direction='in', labelsize=35, size=2, width=3, pad = 15)
ax2.tick_params(axis='both', which='minor', direction='in', labelsize=35, size=2, width=3, pad = 15)
ax3.tick_params(axis='both', which='minor', direction='in', labelsize=35, size=2, width=3, pad = 15)


ax2.hist(means, bins = 15, density=True, histtype='step', edgecolor='b', linewidth=4, facecolor='lightblue', hatch='/', fill=True)
ax2.vlines(x=bias_0, ymin=0, ymax=0.5, lw=4, color='magenta', label=r'$\langle \hat{{H}}_0\rangle={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax3.hist(stds, bins = 15, density=True, histtype='step', linewidth=4, edgecolor='b', facecolor='lightblue', hatch='/',  fill=True)
ax3.vlines(x=std_u, ymin=0, ymax=4.5, lw=4, color='magenta', label=r'$\hat{{\sigma}}_{{H_0}}={:.2f}\pm{:.2f}$'.format(std_u, std_std))

ax2.legend(fontsize=30, loc='upper right', framealpha=1)
ax3.legend(fontsize=30, loc='upper right', framealpha=1)
ax1.set_xlim(60,80)
ax1.set_ylim(0,0.52)
ax2.set_ylim(0,0.42)
#ax3.set_ylim(0,3.2)
ax3.set_ylim(0,4.3)
ax3.set_xlim(0.5,2.2)

ax1.set_xlabel(r'$ H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=40, labelpad=15)
ax2.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=40, labelpad=15)
ax3.set_xlabel(r'$\sigma_{H_0}$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=40, labelpad=15)

ax1.set_ylabel(r'$ P\,(H_0|\boldsymbol{d_{\mathrm{GW}}})$',fontsize=40, labelpad=20)
ax2.set_ylabel(r'$P\,(\hat{H}_0|\mathrm{samples})$',fontsize=40, labelpad=20)
ax3.set_ylabel(r'$P\,(\sigma_{H_0}|\mathrm{samples})$',fontsize=40, labelpad=20)

plt.show()
print(bias_0)
print(bias_err_0)
print(np.mean(stds))

# %%


from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator

#plt.style.use('default')

# Single event posterior plot
i = 0
filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
df = pd.read_csv(filename, index_col = 0)
dfp = df.prod(axis=1)
dfp /= dfp.sum()*(dfp.index[1]-dfp.index[0])



fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ymax=0.4
cmap = cm.get_cmap('winter')
color = iter(cm.winter(np.linspace(0, 1, len(df.columns))))
for i in range(len(df.columns)):
    c = next(color)
    ax.plot(df.index, df[str(i)], c=c, alpha=0.75)
    if i == int(len(df.columns)/3.01):
        mid_c = c
ax.plot([],[], c = mid_c, alpha=0.5, label='Single event posteriors')
ax.plot(dfp.index, dfp.values, c='magenta', lw=5, label='Full posterior')
ax.vlines(x=70, ymin=0, ymax=ymax, color='r', lw=3, ls='dashed', label='True value')


ax.set_xlim(50,100)
ax.set_ylim(0,ymax)
ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.tick_params(axis='both', which='major', direction='in', labelsize=35, size=10, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$P\,(\,H_0\, |\, d_{\mathrm{GW}}\,)$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)

#ax.set_xlim(59.9,80.1)
ax.set_ylim(-0.002,0.4)
ax.tick_params(axis='both', which='minor', direction='in', size=5, width=3, pad = 9)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//single_event.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()