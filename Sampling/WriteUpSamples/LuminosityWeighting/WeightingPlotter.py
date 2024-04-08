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
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

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


#%%

investigated_characteristic = 'D_max_ratio'
investigated_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
selection_accounted = [True, False]
r_or_w = ['right', 'wrong']
max_numbers = ["0" for i in range(len(investigated_values))]

#%%


def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

meanss_s = []
stdss_s = []
pos_s = []

p_i_s_s = []
c_i_s_s = []

for k in range(len(selection_accounted)):

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
        filename = "SampleUniverse_"+str(investigated_characteristic)+r_or_w[k]+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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

    meanss_s.append(meanss)
    stdss_s.append(stdss)
    pos_s.append(pos)
    p_i_s_s.append(p_i_s)
    c_i_s_s.append(c_i_s)

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
    
N = 50
ci = np.linspace(1/N,1,N)

for i in range(len(investigated_values)):
    fraction = []
    for j in ci:    
        fraction.append(sum(k<j for k in c_i_s[i])/len(c_i_s[i]))
    plt.stairs(fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])

plt.legend()
plt.show()

#%%

for j in range(len(selection_accounted)):
    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot()
    B = 10
    bins = np.linspace(0,1,B+1)
    NN = len(p_i_s[0])
    u_p = 0.995
    d_p = 0.005

    for i in range(len(investigated_values)):
        ax.hist(p_i_s_s[j][i], bins=bins, density=0, histtype='step', lw=3, label=investigated_values[i])

    upper_band = sps.binom.ppf(u_p, NN, 1/B)
    lower_band = sps.binom.ppf(d_p, NN, 1/B)

    ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='r', ls='dashed')
    ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='coral', alpha=0.5)
    ax.set_title('{} Average Events'.format(selection_accounted[j]))
    plt.legend()
    plt.show()

#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

biases_s = []
biases_err_s = []
for j in range(len(selection_accounted)):    
    biases = []
    biases_err = []
    for i in range(len(investigated_values)):
        bias, bias_err = expected(meanss_s[j][i], stdss_s[j][i])    
        s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
        print(j)
        print(s)        
        biases.append(bias)
        biases_err.append(bias_err)    
    biases_s.append(biases)
    biases_err_s.append(biases_err)

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(selection_accounted))))
for j in range(len(selection_accounted)):
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(investigated_values), np.array(biases_s[j])-70, marker='^', s=100, c=c, label=r'Modelled = {}'.format(str(selection_accounted[j])), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(investigated_values), np.array(biases_s[j])-70, c=c, zorder=2)
    ax.errorbar(np.array(investigated_values), np.array(biases_s[j])-70, yerr=np.array(biases_err_s[j]), capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\langle\hat{H_0} - H_0\rangle$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$D_{\max}/S$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


#%%


sigmas_s = []
sigmas_unc_s = []

for j in range(len(selection_accounted)):        
    sigmas = []
    sigmas_unc = []
    for i in range(len(investigated_values)):
        sigmas.append(np.mean(stdss_s[j][i]))
        sigmas_unc.append(np.std(stdss_s[j][i])/np.sqrt(len(stdss_s[j][i])))
    sigmas_s.append(sigmas)
    sigmas_unc_s.append(sigmas_unc)


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(selection_accounted))))
for j in range(len(selection_accounted)):
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(investigated_values), np.array(sigmas_s[j])/70, marker='^', s=100, c=c, label=r'Modelled = {}'.format(str(selection_accounted[j])), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(investigated_values), np.array(sigmas_s[j])/70, c=c, zorder=2)
    ax.errorbar(np.array(investigated_values), np.array(sigmas_s[j])/70, yerr=np.array(sigmas_unc_s[j])/70, capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}/H_0$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$D_{\max}/S$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%





















#%%


investigated_characteristic = 'lum_weighting'
investigated_values = ['Proportional', 'Proportional', 'Random', 'Random']
investigated_values_inference = ['Proportional', 'Random', 'Random', 'Proportional']
max_numbers = ["0" for i in range(len(investigated_values))]

#b = []
#f = []
    
Ns = [5, 10, 20, 50, 100, 200]

#%%

investigated_characteristic = 'lum_weighting_pres'
investigated_values = ['Proportional', 'Proportional', 'Random', 'Random']
investigated_values_inference = ['Proportional', 'Random', 'Random', 'Proportional']
max_numbers = ["0" for i in range(len(investigated_values))]

Ns = [2,4,8,16,32,64,128]

#%%

meanss_s = []
stdss_s = []
pos_s = []

p_i_s_s = []
c_i_s_s = []

for j in Ns:
    investigated_characteristic = 'lum_weighting' + '_' + str(j) + '_' + 'average_events'

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
        if investigated_values[i] == investigated_values_inference[i]:
            r_or_w = 'right'
        else:
            r_or_w = 'wrong'
        #print(i)
        filename = "SampleUniverse_"+str(investigated_characteristic)+r_or_w+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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

    meanss_s.append(meanss)
    stdss_s.append(stdss)
    pos_s.append(pos)
    p_i_s_s.append(p_i_s)
    c_i_s_s.append(c_i_s)
    

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    #'''
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

    plt.show()


#%%

N = 500
ci = np.linspace(1/N,1,N)

for i in range(len(investigated_values)):
    fraction = []
    for j in ci:    
        fraction.append(sum(k<j for k in c_i_s[i])/len(c_i_s[i]))
    plt.stairs(fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])

plt.legend()
plt.show()

#%%

for j in range(len(Ns)):
    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot()
    B = 10
    bins = np.linspace(0,1,B+1)
    NN = len(p_i_s[0])
    u_p = 0.995
    d_p = 0.005

    for i in range(len(investigated_values)):
        ax.hist(p_i_s_s[j][i], bins=bins, density=0, histtype='step', lw=3, label=investigated_values[i])

    upper_band = sps.binom.ppf(u_p, NN, 1/B)
    lower_band = sps.binom.ppf(d_p, NN, 1/B)

    ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='r', ls='dashed')
    ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='coral', alpha=0.5)
    ax.set_title('{} Average Events'.format(Ns[j]))
    plt.legend()
    plt.show()

#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

biases_s = []
biases_err_s = []
for j in range(len(Ns)):    
    biases = []
    biases_err = []
    for i in range(len(investigated_values)):
        bias, bias_err = expected(meanss_s[j][i], stdss_s[j][i])    
        s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
        print(j)
        print(s)        
        biases.append(bias)
        biases_err.append(bias_err)    
    biases_s.append(biases)
    biases_err_s.append(biases_err)

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)):
    if investigated_values[j]==investigated_values_inference[j]:    
        correct = 'correctly'
    else:
        correct = 'incorrectly'
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(Ns), np.array(biases_s)[:,j]-70, marker='^', s=100, c=c, label=r'{} modelled {}'.format(investigated_values[j],correct), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(Ns), np.array(biases_s)[:,j]-70, c=c, zorder=2)
    ax.errorbar(np.array(Ns), np.array(biases_s)[:,j]-70, yerr=np.array(biases_err_s)[:,j], capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\langle\hat{H_0} - H_0\rangle$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar N$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


#%%

sigmas_s = []
sigmas_unc_s = []

for j in range(len(Ns)):        
    sigmas = []
    sigmas_unc = []
    for i in range(len(investigated_values)):
        sigmas.append(np.mean(stdss_s[j][i]))
        sigmas_unc.append(np.std(stdss_s[j][i])/np.sqrt(len(stdss_s[j][i])))
        print(len(stdss_s[j][i]))
    sigmas_s.append(sigmas)
    sigmas_unc_s.append(sigmas_unc)


# %%

x = np.linspace(4.5,210,1000)
x = np.linspace(1.5,160,1000)
y = 4.5/(70*np.sqrt(x))
y = 5.5/(70*np.sqrt(x))

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)):
    if investigated_values[j]==investigated_values_inference[j]:    
        correct = 'correctly'
    else:
        correct = 'incorrectly'
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(Ns), np.array(sigmas_s)[:,j]/70, marker='^', s=100, c=c, label=r'{} modelled {}'.format(investigated_values[j], correct), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(Ns), np.array(sigmas_s)[:,j]/70, c=c, zorder=2)
    ax.errorbar(np.array(Ns), np.array(sigmas_s)[:,j]/70, yerr=np.array(sigmas_unc_s)[:,j]/70, capsize=5, c=c, fmt='None', zorder=1)

ax.plot(x,y,ls='dashed', c='r', label=r'$\propto \bar N\,^{-1/2}$')
ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 23, loc='lower left', framealpha=1)
ax.set_ylabel(r'$\hat \sigma_{H_0}/H_0$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar N$', fontsize=35, labelpad=15)
ax.set_ylim(0.003,0.15)

#ax.set_xlim(4,1100)
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()



#%%

















# %%

#investigated_values = [2.964, 3.409, 4.953, 8.827] + [4.035, 6.382, 13.804, 28.878] + [2.637, 2.3902]
#investigated_values.sort(reverse=True)
#rel = [40, 35, 25, 15] + [5, 10, 20, 30] + [45, 50]

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

def sel_eff(x,D,c,k):
    return 1 - (1 + (D/(x*300000/70))**c)**(-k)

cs = [4.035, 6.382, 13.804, 28.878]
cs.sort(reverse=True)
unc = [5, 10, 20, 30]
x = np.linspace(0,0.15,10000)


color = iter(cm.cool(np.linspace(0, 1, len(investigated_values))))
for i in range(len(cs)):
    y = sel_eff(x, 250, cs[i], 2)
    c = next(color)
    ax.plot(x,y,ls='dashed', lw=5, c=c, label=r'$\sigma_D/D = {}\%$'.format(unc[i]))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

#ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, loc='upper right', framealpha=1)
ax.set_ylabel(r'$P_{\mathrm{det}}(D_{\mathrm{GW}}|z,H_0)$', fontsize=45, labelpad=15)
ax.set_xlabel(r'$z$', fontsize=45, labelpad=15)
#ax.set_ylim(0.003,0.15)
#ax.set_xlim(4,1100)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%

def burr(x,D,c,k):
    return (c*k/D)*((x/D)**(c-1))/((1+(x/D)**c)**(k+1))

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

x = np.linspace(0,200,10000)

color = iter(cm.cool(np.linspace(0, 1, 4)))
for i in range(len(cs)):
    y = burr(x, 100, cs[i], 2)
    c = next(color)
    ax.plot(x,y,ls='solid', lw=5, c=c, label=r'$\sigma_D/D = {}\%$'.format(unc[i]))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

ax.vlines(x=100, ymin=-0.005,ymax=0.1, color='r', ls='dashed', lw=4)
ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
ax.set_ylim(-0.0005,0.1)
ax.set_xlim(0,200)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 28, loc='upper right', framealpha=1)
ax.set_ylabel(r'$P\,(\hat{D}|D)$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$D$ (Mpc)', fontsize=35, labelpad=15)
#ax.set_ylim(0.003,0.15)
#ax.set_xlim(4,1100)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()



# %%
