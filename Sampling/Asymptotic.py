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

#%%

investigated_characteristic = 'event_num_log_powerpoint_standard'
investigated_values = [5, 10, 20, 40, 80, 160]
investigated_values = [2, 4, 8, 16, 32, 64, 128, 256]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 200, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_gal_n = 5000, specify_gal_number = True, wanted_det_events = investigated_values[i], specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


# %%

investigated_characteristic = 'event_num_log_powerpoint_standard'
investigated_values = [5, 10, 20, 40, 80, 160]
investigated_values = [2, 4, 8, 16, 32, 64, 128, 256]
max_numbers = ['0']*len(investigated_values)

#%%

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

# /Users/daneverett/PycharmProjects/MSciProject/Sampling/PosteriorData/SampleUniverse_event_num_log_powerpoint_standard_2_0.csv

for i in range(len(investigated_values)):
    #print(i)
    filename = "/SamplingPosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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
ax.set_xlabel(r'$\bar{N}$', fontsize=35, labelpad=15)
ax.set_xscale('log')
ax.set_xticks(investigated_values)
ax.set_xticklabels([r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
       
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

ax.scatter(investigated_values, sigmas/70, marker='^', s=100, c='b', label='Data', zorder=2)
ax.plot(investigated_values, sigmas/70, marker='^', c='dodgerblue', zorder=1)
ax.errorbar(investigated_values, sigmas/70, yerr=sigmas_unc/70, capsize=5, c='dodgerblue', fmt='None', zorder=0)

ax.plot(x, func(x, *popt), ls='dashed', dashes=(5,5), lw=3, c='magenta', label=r'$\frac{{\hat{{\sigma}}_{{H_0}}}}{{H_0}} = \frac{{\alpha}}{{\sqrt{{\bar{{N}}}}}}$ fit, $\alpha={:.1f}\%\pm{:.1f}\%$'.format(100*popt[0]/70, 100*pcov[0,0]**0.5/70))


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
                
ax.legend(fontsize=27, loc='upper right')          
plt.show()











# %%








#%%

title = 'Posterior Asymptotic Normality & Constraining Power'

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


#%%

import random

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

# if linear
dist = 10
dist=0
cap = 5
h_space = N[1]-N[0]-dist

outer_x_min = 60
outer_x_max = 82.5
#outer_x_min = 67
#outer_x_max = 73
plot_lim_min = 10
plot_lim_max = 320
plot_lim_min = 1.7
rate = N[-1]/N[-2]
plot_lim_max = rate*N[-1] + 1

ax.set_ylim(outer_x_min,outer_x_max)
ax.set_xlim(plot_lim_min, plot_lim_max)

mus = []
sigs = []
maxs = []
argmaxs = []

space = N.tolist()
space.append(N[-1]*2)

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

axs = []
for i in range(len(df_N.columns)):
    ax1 = ax.inset_axes(
        [N[i], outer_x_min, (space[i+1]-space[i])/1.5, (outer_x_max-outer_x_min)], transform=ax.transData, zorder=1)
    axs.append(ax1)
for i in range(len(df_N.columns)):
  
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    df.dropna(inplace=True, axis=1)
    pdf_single = df[column]/df[column].sum() #* (df.index[1] - df.index[0])
    pdf_single.dropna(inplace=True)
    #vals = np.array(pdf_single.index)
    #mean = sum(pdf_single*vals)
    axs[i].spines[['right', 'top', 'bottom','left']].set_visible(False)   
    n = random.choice(df.columns)
    axs[i].plot(df[n], df.index, c='blue', lw=3.5)
    
    #axs[i].plot(df_N[df_N.columns[i]], df_N.index, c='turquoise', lw=3.5)
    
    #axs[i].plot(df_N[df_N.columns[i]], df_N.index, c='turquoise', lw=3.5)
    
    pdf = df_N[df_N.columns[i]]
    vals = np.array(pdf_single.index)
    max_p = np.max(pdf)
    argmax_p = vals[np.argmax(pdf)]
    mu = sum(pdf*vals)
    sig = np.sqrt(sum((pdf*vals**2))-mu**2)
    
    mu,_ = expected(meanss[i], stdss[i])
    sig = np.mean(stdss[i])
    
    mus.append(mu)
    sigs.append(sig)
    maxs.append(max_p)
    argmaxs.append(argmax_p)
    adj = 1/(df_N.index[1] - df_N.index[0])
    #h = np.exp(-0.5)/(adj*np.sqrt(2*np.pi)*sig)
    #axs[i].vlines(x=h, ymax= mu+sig, ymin = mu-sig)
    #axs[i].hlines(y = [mu-sig,mu, mu+sig], xmax=h+h/2, xmin=h-h/2)
    #ax.vlines(x=N[i], ymax= mu+sig, ymin = mu-sig, lw=2, color='black')
    #ax.hlines(y = [mu-sig,mu, mu+sig], xmax=N[i]+(space[i+1]-space[i])/8, xmin=N[i], lw=2, color='black')
    axs[i].set_ylim(outer_x_min,outer_x_max)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    #axs[i].set_xlim(0,120)

x = N
y = sigs
x_cont = np.linspace(N[0]/rate,N[-1]*rate,1000)

#def func(x,a):
#    return a/np.sqrt(x)

#popt, pcov = curve_fit(func, x, y)

ax.plot(x_cont, 70 + 70*func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r', label=r'$\pm 1\,\sigma = H_0\,\alpha/\sqrt{N}$')#, label=r'$\sigma_{{H_0}} = \frac{{\alpha}}{{\sqrt{{N}}}}$ fit, $\alpha={:.1f}\%\pm{:.1f}\%$'.format(100*popt[0]/70, 100*pcov[0,0]**0.5/70))
ax.plot(x_cont, 70 - 70*func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r')
ax.plot([], [], c='blue', label='Posterior examples')
#ax.plot([], [], c='white', label=r'$\mu\pm 1\sigma$')

ax.hlines(y=70, xmin=plot_lim_min, xmax=plot_lim_max, color='magenta', ls='dashed', lw=3)

ax.set_xscale('log')
N_labels = [str(i) for i in N]
ax.set_xticks(N)
ax.set_xticklabels(N_labels)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax.grid(axis='y', ls='dashed', alpha=0.7)
ax.set_xlabel('Average number of detected mergers', fontsize=45, labelpad=15)
ax.set_xlabel(r'$\bar{N}$', fontsize=45, labelpad=15)

ax.set_ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
#ax.set_title(title, x=0.46, fontsize=35, pad=30)
#ax.legend(fontsize=28, framealpha=1, loc=(0.357,0.705))
ax.legend(fontsize=27, framealpha=1)#, loc=(0.342,0.715))
ax.set_xticks(investigated_values)
ax.set_xticklabels([r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'], fontsize=40)
   

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//asymptotic_normality.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()
# %%
