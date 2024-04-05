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

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

#%%


investigated_characteristic = 'event_num_log_powerpoint'
investigated_values = [5, 10, 20, 40, 80, 160]
investigated_values = [2, 4, 8, 16, 32, 64, 128, 256]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 200, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_gal_n = 2000, specify_gal_number = True, wanted_det_events = investigated_values[i], specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


# %%


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

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title


#%%



fig = plt.figure(figsize = (12,8))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=2,
                        wspace=0.2,
                         hspace=0.3)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])

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

ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
'''
ax1.violinplot(meanss, bw_method=0.4, vert=False, showmeans=True)
ax1.set_yticks(pos)
ax1.set_yticklabels(rel, fontsize=20)
ax1.set_title('Means', fontsize = 25)
ax1.grid(axis='x')

ax2.violinplot(stdss, vert=False, showmeans=True)
ax2.set_yticks(pos)
ax2.set_yticklabels(rel, fontsize=20)
ax2.set_title('Standard deviations', fontsize = 25)
fig.supylabel(axis_namer(investigated_characteristic), fontsize=20)
ax2.grid(axis='x')
'''
ax1.violinplot(meanss, bw_method=0.4, vert=False, showmeans=True)
ax1.set_yticks(pos)
ax1.set_yticklabels(investigated_values, fontsize=20)
ax1.set_title('Means', fontsize = 25)
ax1.grid(axis='x')

ax2.violinplot(stdss, vert=False, showmeans=True)
ax2.set_yticks(pos)
ax2.set_yticklabels(investigated_values, fontsize=20)
ax2.set_title('Standard deviations', fontsize = 25)
fig.supylabel(axis_namer(investigated_characteristic), fontsize=20)
ax2.grid(axis='x')
#'''
plt.show()

#%%

N = 500
ci = np.linspace(1/N,1,N)

'''
significance = [1,2,3]
for i in range(len(significance)): 
    df_sig = pd.DataFrame(index=df.index)
    for j in range(100):
        #h = np.random.uniform(65,75)# + 5*significance[i])
        h = 70 + 5*significance[i]
        df_sig[str(j)] = sps.norm.pdf(df.index, loc=h, scale=5)

c_i_s_signif = C_I_samp(df_sig)
'''

for i in range(len(investigated_values)):
    fraction = []
    for j in ci:    
        fraction.append(sum(k<j for k in c_i_s[i])/len(c_i_s[i]))
    plt.stairs(fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])

'''
sigs_fraction = []
for j in ci:    
        sigs_fraction.append(sum(k<j for k in c_i_s_signif)/len(c_i_s_signif))
plt.stairs(sigs_fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])
'''

plt.legend()
plt.show()

#%%

B = 10
bins = np.linspace(0,1,B+1)
NN = len(p_i_s[0])
u_p = 0.995
d_p = 0.005

for i in range(len(investigated_values)):
    plt.hist(p_i_s[i], bins=bins, density=0, histtype='step', lw=3, label=investigated_values[i])

upper_band = sps.binom.ppf(u_p, NN, 1/B)
lower_band = sps.binom.ppf(d_p, NN, 1/B)

plt.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='r', ls='dashed')
plt.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='coral', alpha=0.5)
plt.legend()
plt.show()

#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

biases = []
biases_err = []
for i in range(len(investigated_values)):
    bias, bias_err = expected(meanss[i], stdss[i])    
    s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
    print(s)        
    biases.append(bias)
    biases_err.append(bias_err)


#%%
    
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()


c = np.array([0,0,1,1])
edgecolor='white'
ax.scatter(np.array(investigated_values), np.array(biases)-70, marker='^', s=100, c=c, zorder=3)
c = 'dodgerblue'
ax.plot(np.array(investigated_values), np.array(biases)-70, c=c, zorder=2)
ax.errorbar(np.array(investigated_values), np.array(biases)-70, yerr=np.array(biases_err), capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\langle\hat{H_0} - H_0\rangle$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\sigma_z$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


#%%
    
x = np.linspace(1.5,350,1000)
y = 4.5/(70*np.sqrt(x))

sigmas = []
sigmas_unc = []
for i in range(len(investigated_values)):
    sigmas.append(np.mean(stdss[i]))
    sigmas_unc.append(np.std(stdss[i]))

sigmas = np.array(sigmas)
sigmas_unc = np.array(sigmas_unc)


def func(x,a):
    return a/(70*np.sqrt(x))

popt, pcov = curve_fit(func, investigated_values, sigmas/70, sigma=sigmas_unc/70)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.plot(x, func(x, *popt), ls='dashed', dashes=(5,5), lw=3, c='magenta', label=r'$\hat{{\sigma}}_{{H_0}}/H_0 = \alpha/\sqrt{{\bar{{N}}}}$ fit, $\alpha={:.1f}\%\pm{:.1f}\%$'.format(100*popt[0]/70, 100*pcov[0,0]**0.5/70))

ax.scatter(investigated_values, sigmas/70, marker='^', s=100, c='b', label='Data', zorder=2)
ax.plot(investigated_values, sigmas/70, marker='^', c='dodgerblue', zorder=1)
ax.errorbar(investigated_values, sigmas/70, yerr=sigmas_unc/70, capsize=5, c='dodgerblue', fmt='None', zorder=0)

#ax.plot(x,y,ls='dashed', c='r', label=r'$\propto \bar N\,^{-1/2}$')

ax.grid(ls='dashed', c='lightblue', axis='y', which='minor', alpha=1)
ax.grid(ls='dashed', c='lightblue', axis='x', which='major', alpha=1)

#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\hat{{\sigma}}_{H_0}/H_0$', fontsize=45, labelpad=15)
ax.set_xlabel(r'$\bar{N}$', fontsize=45, labelpad=15)
ax.set_xscale('log')
ax.set_yscale('log')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
ax.tick_params(axis='both', top=True, right=True, which='major', direction='in', labelsize=42, size=9, width=3, pad = 15)
ax.tick_params(axis='y', which='minor', direction='in', labelsize=35, size=4, width=3, pad = 15)

ax.set_xticks(investigated_values)
ax.set_xticklabels([r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
                
ax.legend(fontsize=25, loc='lower left')          
plt.show()


# %%
