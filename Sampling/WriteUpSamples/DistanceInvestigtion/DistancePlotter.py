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


def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

#%%

investigated_characteristic = 'delta_D'
investigated_values = [2.964, 3.409, 4.953, 8.827] + [4.035, 6.382, 13.804, 28.878] + [2.637, 2.3902]
investigated_values.sort(reverse=True)
rel = [40, 35, 25, 15] + [5, 10, 20, 30] + [45, 50]
rel.sort()
max_numbers = ["0" for i in range(len(investigated_values))]

#%%

investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000, 16000]
max_numbers = ["0" for i in range(len(investigated_values))]


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
    filename = "c:\\Users\manco\OneDrive\Ambiente de Trabalho\Masters_Project\MsciProj\Sampling\PosteriorData\SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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

for i in range(len(investigated_values)):
    bias, bias_err = expected(meanss[i], stdss[i])    
    s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
    print(s)


#%%
    
sigmas = []
sigmas_unc = []
for i in range(len(investigated_values)):
    sigmas.append(np.mean(stdss[i]))
    sigmas_unc.append(np.std(stdss[i])/np.sqrt(len(stdss[i])))

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.scatter(rel, sigmas, marker='^', s=20, c='b')
ax.errorbar(rel, sigmas, yerr=sigmas_unc, capsize=5, c='turquoise', fmt='None')
ax.grid(ls='dashed', c='lightblue', alpha=0.8)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\sigma_D/D$', fontsize=35, labelpad=15)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()

# %%

sigmas = []
sigmas_unc = []
for i in range(len(investigated_values)):
    sigmas.append(np.mean(stdss[i]))
    sigmas_unc.append(np.std(stdss[i])/np.sqrt(len(stdss[i])))


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.scatter(np.array(rel)/100, np.array(sigmas)/70, marker='^', s=100, c='r', zorder=2)
ax.errorbar(np.array(rel)/100, np.array(sigmas)/70, yerr=np.array(sigmas_unc)/70, capsize=5, c='b', fmt='None', zorder=1)
ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}/H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\sigma_D/D$', fontsize=35, labelpad=15)
ax.set_ylim(-0.01,0.2)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.scatter(np.array(investigated_values), np.array(sigmas)/70, marker='^', s=100, c='r', zorder=2)
ax.errorbar(np.array(investigated_values), np.array(sigmas)/70, yerr=np.array(sigmas_unc)/70, capsize=5, c='b', fmt='None', zorder=1)
ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}/H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'Number of Galaxies', fontsize=35, labelpad=15)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
ax.set_xscale('log')
plt.show()

#%%





















# %%


investigated_characteristic = 'delta_D'
investigated_values = [2.964, 3.409, 4.953, 8.827] + [4.035, 6.382, 13.804, 28.878] #+ [2.637, 2.3902]
investigated_values.sort(reverse=True)
rel = [40, 35, 25, 15] + [5, 10, 20, 30] #+ [45, 50]
rel.sort()
max_numbers = ["0" for i in range(len(investigated_values))]

Ns = [5, 10, 20, 50, 100, 200]

#%%

investigated_characteristic = 'delta_D_standard'
investigated_values = [6.382, 8.827, 13.804, 28.878] + [4.953, 7.423, 10.81, 18.817] + [5.582]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [7.5, 12.5, 17.5] + [5, 10, 15, 20, 25] + [22.5]
rel.sort()
Ns = [2, 4, 8, 16, 32, 64, 128]
max_numbers = ["0" for i in range(len(investigated_values))]
event_count_max_numbers = ['0']*len(investigated_values)

#%%

meanss_s = []
stdss_s = []
pos_s = []

p_i_s_s = []
c_i_s_s = []

# ADDDDDDDDDDD UNIFORM PRIOR FILTERRRRRRRR !!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# DONE

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

for j in Ns:
    investigated_characteristic = 'delta_D' + '_' + str(j) + '_' + 'average_events'

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
        filename = "c:\\Users\manco\OneDrive\Ambiente de Trabalho\Masters_Project\MsciProj\Sampling\PosteriorData\SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
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
    ax1.set_yticklabels(rel, fontsize=20)
    ax1.set_title('Means', fontsize = 25)
    ax1.grid(axis='x')

    ax2.violinplot(stdss, vert=False, showmeans=True)
    ax2.set_yticks(pos)
    ax2.set_yticklabels(rel, fontsize=20)
    ax2.set_title('Standard deviations', fontsize = 25)
    fig.supylabel(axis_namer(investigated_characteristic), fontsize=20)
    ax2.grid(axis='x')

    plt.show()


#%%

meanss_s = []
stdss_s = []
pos_s = []

p_i_s_s = []
c_i_s_s = []

Ns_meanss = []
# ADDDDDDDDDDD UNIFORM PRIOR FILTERRRRRRRR !!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# DONE

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

for j in Ns:
    investigated_characteristic = 'delta_D_standard' + '_' + str(j) + '_' + 'average_events'

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
    Ns_means = []


    for i in range(len(investigated_values)):
        #print(i)
        filename = "SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
        df = pd.read_csv(filename, index_col = 0)
        means = []
        stds = []
        inc = df.index[1]-df.index[0]
        p_i_s.append(bias_dist(df))
        c_i_s.append(C_I_samp(df))
        filename = "EventCount_SampleUniverse_" + str(investigated_characteristic) + "_" +str(investigated_values[i]) + "_" + \
                   event_count_max_numbers[i] + ".csv"
        df_event_count = pd.read_csv(filename, index_col=0)
        df_event_count.dropna(inplace=True, axis=1)
        

        Ns_means_no_zero = []
        for column in df.columns:
            if is_unique(df[column]):
                print('Gotcha')
                continue
            

            Ns_means_no_zero.append(df_event_count[column])
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
        Ns_means.append(np.mean(Ns_means_no_zero))

    meanss_s.append(meanss)
    stdss_s.append(stdss)
    pos_s.append(pos)
    p_i_s_s.append(p_i_s)
    c_i_s_s.append(c_i_s)
    Ns_meanss.append(Ns_means)
    

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    #'''
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

for j in range(len(Ns)):
    for i in range(len(investigated_values)):
        bias, bias_err = expected(meanss_s[j][i], stdss_s[j][i])    
        s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
        print(j)
        print(s)

#%%

sigmas_s = []
sigmas_unc_s = []

for j in range(len(Ns)):        
    sigmas = []
    sigmas_unc = []
    for i in range(len(investigated_values)):
        sigmas.append(np.mean(stdss_s[j][i]))
        sigmas_unc.append(np.std(stdss_s[j][i])/np.sqrt(len(stdss_s[j][i])))
    sigmas_s.append(sigmas)
    sigmas_unc_s.append(sigmas_unc)


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(Ns))))
for j in range(len(Ns)):
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(rel)/100, np.array(sigmas_s[j])/70, marker='^', s=100, c=c, label=r'${}$'.format(Ns[j]), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(rel)/100, np.array(sigmas_s[j])/70, c=c, zorder=2)
    ax.errorbar(np.array(rel)/100, np.array(sigmas_s[j])/70, yerr=np.array(sigmas_unc_s[j])/70, capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='cadetblue', alpha=0.7, zorder=0)
#ax.set_xlim(0.035,0.3)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(title=r'$\bar{N}$', title_fontsize=30, fontsize = 30, framealpha=1, loc='upper center', handletextpad=0.01, columnspacing=0.01, ncol=len(Ns))
ax.set_ylabel(r'$\sigma_{H_0}/H_0$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\sigma_D/D$', fontsize=35, labelpad=15)
ax.set_ylim(-0.01,0.27)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)-1,-1,-1):
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, marker='^', s=100, c=c, label=r'${}\% $'.format(rel[j]), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, c=c, zorder=2)
    ax.errorbar(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, yerr=np.array(sigmas_unc_s)[:,j]/70, capsize=5, c=c, fmt='None', zorder=1)

#ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.grid(ls='dashed', c='cadetblue', axis='y', which='both', alpha=0.7)
ax.grid(ls='dashed', c='cadetblue', axis='x', which='major', alpha=0.7)

ax.tick_params(axis='both', which='major', direction='in', labelsize=35, size=8, width=3, pad = 15)
ax.tick_params(axis='y', which='minor', direction='in', labelsize=35, size=4, width=3, pad = 15)

ax.legend(title=r'$\sigma_D/D$',title_fontsize=30,fontsize = 25, loc='lower right', framealpha=1)
ax.set_ylabel(r'$\hat{\sigma}_{H_0}/H_0$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar{N}$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
ax.set_xlim(1.5,700)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(Ns)
ax.set_xticklabels([r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$', r'$2^6$', r'$2^7$'])

#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%

def func(x,a):
    return a/(np.sqrt(x))

alphas = []
alphas_std = []
for j in range(len(investigated_values)):
    popt, pcov = curve_fit(func, np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, sigma=np.array(sigmas_unc_s)[:,j]/70)
    alphas.append(popt[0])
    alphas_std.append(np.sqrt(pcov[0,0]))

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.scatter(np.array(rel)/100, np.array(alphas), marker='^', s=100, c='r', zorder=3)
c[3] = 0.7
ax.plot(np.array(rel)/100, np.array(alphas), c='r', zorder=2)
ax.errorbar(np.array(rel)/100, np.array(alphas), yerr=np.array(alphas_std), capsize=5, c='r', fmt='None', zorder=1)

ax.grid(ls='dashed', c='cadetblue', alpha=0.7, zorder=0)
#ax.set_xlim(0.035,0.3)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax.legend(title=r'$\bar{N}$', title_fontsize=30, fontsize = 27, framealpha=1, loc='upper left', handletextpad=0.01, columnspacing=0.01, ncol=len(Ns))
ax.set_ylabel(r'$\alpha$', fontsize=40, labelpad=15)
ax.set_xlabel(r'$\sigma_D/D$', fontsize=40, labelpad=15)
#ax.set_ylim(-0.01,0.25)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


#%%





#%%


investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000, 16000]
max_numbers = ["0" for i in range(len(investigated_values))]


#%%


Ns = [5, 10, 20, 50, 100, 200]

#%%

investigated_characteristic = 'gal_num_set_standard_proportional'
investigated_values = [500, 1000, 2000, 4000, 8000]#, 16000]
max_numbers = ["0" for i in range(len(investigated_values))]

#b = []
#f = []
    
Ns = [2, 4, 8, 16, 32, 64, 128]
event_count_max_numbers = ['0']*len(investigated_values)


#%%

investigated_characteristic = 'gal_num_set_standard_random'
investigated_values = [500, 1000, 2000, 4000, 8000]#, 16000]
max_numbers = ["0" for i in range(len(investigated_values))]

#b = []
#f = []
    
Ns = [2, 4, 8, 16, 32, 64, 128]
event_count_max_numbers = ['0']*len(investigated_values)

#%%

max_numbers = ["1" for i in range(len(investigated_values))]


#%%

meanss_s = []
stdss_s = []
pos_s = []

p_i_s_s = []
c_i_s_s = []

Ns_meanss = []

# ADDDDDDDDDDD UNIFORM PRIOR FILTERRRRRRRR !!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# DONE

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

for j in Ns:
    investigated_characteristic = 'gal_num_set' + '_' + str(j) + '_' + 'average_events'

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
    
    Ns_means = []

    for i in range(len(investigated_values)):
        if investigated_values[i] == 16000 and (j == 100 or j == 200):
            continue
        #print(i)
        filename = "SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
        df = pd.read_csv(filename, index_col = 0)
        means = []
        stds = []
        inc = df.index[1]-df.index[0]
        p_i_s.append(bias_dist(df))
        c_i_s.append(C_I_samp(df))

        filename = "EventCount_SampleUniverse_" + str(investigated_characteristic) + "_" +str(investigated_values[i]) + "_" + \
                   event_count_max_numbers[i] + ".csv"
        df_event_count = pd.read_csv(filename, index_col=0)
        df_event_count.dropna(inplace=True, axis=1)
        
        Ns_means_no_zero = []

        for column in df.columns:
            if is_unique(df[column]):
                print('Gotcha')
                continue
            
            Ns_means_no_zero.append(df_event_count[column])

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

        Ns_means.append(np.mean(Ns_means_no_zero))

    #if investigated_values[i] == 16000 and (j == 100 or j == 200):
    #    continue
    meanss_s.append(meanss)
    stdss_s.append(stdss)
    pos_s.append(pos)
    p_i_s_s.append(p_i_s)
    c_i_s_s.append(c_i_s)
    
    Ns_meanss.append(Ns_means)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    #'''
    ax1.violinplot(meanss, bw_method=0.4, vert=False, showmeans=True)
    ax1.set_yticks(pos)
    if investigated_values[i] == 16000 and (j == 100 or j == 200):
        ax1.set_yticklabels(investigated_values[:-1], fontsize=20)
    else:
        ax1.set_yticklabels(investigated_values, fontsize=20)
    ax1.set_title('Means', fontsize = 25)
    ax1.grid(axis='x')

    ax2.violinplot(stdss, vert=False, showmeans=True)
    ax2.set_yticks(pos)
    if investigated_values[i] == 16000 and (j == 100 or j == 200):    
        ax2.set_yticklabels(investigated_values[:-1], fontsize=20)
    else:
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
        if investigated_values[i] == 16000 and (Ns[j] == 100 or Ns[j] == 200):
            continue
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

for j in range(len(Ns)):
    for i in range(len(investigated_values)):
        if investigated_values[i] == 16000 and (Ns[j] == 100 or Ns[j] == 200):
            continue
        bias, bias_err = expected(meanss_s[j][i], stdss_s[j][i])    
        s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
        print(j)
        print(s)

#%%

sigmas_s = []
sigmas_unc_s = []

for j in range(len(Ns)):       
    sigmas = []
    sigmas_unc = []
    for i in range(len(investigated_values)):
        if investigated_values[i] == 16000 and (Ns[j] == 100 or Ns[j] == 200):
            continue
        sigmas.append(np.mean(stdss_s[j][i]))
        sigmas_unc.append(np.std(stdss_s[j][i])/np.sqrt(len(stdss_s[j][i])))
    sigmas_s.append(sigmas)
    sigmas_unc_s.append(sigmas_unc)


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(Ns))))
for j in range(len(Ns)):
    c = next(color)
    edgecolor='white'
    if Ns[j] == 100 or Ns[j] == 200:
        ax.scatter(np.array(investigated_values[:-1]), np.array(sigmas_s[j])/70, marker='^', s=100, c=c, label=r'$\bar N = {}$'.format(Ns[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(investigated_values[:-1]), np.array(sigmas_s[j])/70, c=c, zorder=2)
        ax.errorbar(np.array(investigated_values[:-1]), np.array(sigmas_s[j])/70, yerr=np.array(sigmas_unc_s[j])/70, capsize=5, c=c, fmt='None', zorder=1)
    else:
        ax.scatter(np.array(investigated_values)/(8*625**3)*10**6, np.array(sigmas_s[j])/70, marker='^', s=100, c=c, label=r'${}$'.format(Ns[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(investigated_values)/(8*625**3)*10**6, np.array(sigmas_s[j])/70, c=c, zorder=2)
        ax.errorbar(np.array(investigated_values)/(8*625**3)*10**6, np.array(sigmas_s[j])/70, yerr=np.array(sigmas_unc_s[j])/70, capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='cadetblue', alpha=0.7, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
#ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 15)

#ax.legend(fontsize = 28, framealpha=1)
ax.legend(title=r'$\bar{N}$', title_fontsize=30, fontsize = 30, framealpha=1, loc='upper center', handletextpad=0.01, columnspacing=0.01, ncol=len(Ns))
ax.set_ylabel(r'$\sigma_{H_0}/H_0$', fontsize=35, labelpad=15)
#ax.set_xlabel(r'Number of galaxies', fontsize=35, labelpad=15)
ax.set_xlabel(r'$n_*\times 10^{-6}$ Mpc$^{-3}$', fontsize=35, labelpad=15)

#ax.set_ylim(0.,0.179)
ax.set_ylim(0.,0.149)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)-1,-1,-1):
    c = next(color)
    edgecolor='white'
    ax.scatter(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, marker='^', s=100, c=c, label=r'${:.2f}$'.format(10**6 * investigated_values[j]/(8*625**3)), zorder=3)
    c[3] = 0.7
    ax.plot(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, c=c, zorder=2)
    ax.errorbar(np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, yerr=np.array(sigmas_unc_s)[:,j]/70, capsize=5, c=c, fmt='None', zorder=1)

#ax.grid(ls='dashed', c='cadetblue', alpha=0.7, zorder=0)

ax.grid(ls='dashed', c='cadetblue', axis='y', which='both', alpha=0.7)
ax.grid(ls='dashed', c='cadetblue', axis='x', which='major', alpha=0.7)

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 15)
ax.tick_params(axis='y', which='minor', direction='in', labelsize=30, size=4, width=3, pad = 15)

#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
#ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(title=r'$n_*\times 10^{-6}$ Mpc$^{-3}$',title_fontsize=30,fontsize = 25, loc='upper right', handletextpad=0.5, framealpha=1)
ax.set_ylabel(r'$\hat{\sigma}_{H_0}/H_0$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar{N}$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
ax.set_xlim(1.5,200)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(Ns)
ax.set_xticklabels([r'$2^1$',r'$2^2$',r'$2^3$',r'$2^4$',r'$2^5$', r'$2^6$', r'$2^7$'])

#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%

def func(x,a):
    return a/(np.sqrt(x))

alphas = []
alphas_std = []
for j in range(len(investigated_values)):
    #if j==0:
    #    popt, pcov = curve_fit(func, np.array(Ns_meanss)[:-1,j], np.array(sigmas_s)[:-1,j]/70, sigma=np.array(sigmas_unc_s)[:-1,j]/70)
    #    alphas.append(popt[0])
    #    alphas_std.append(np.sqrt(pcov[0,0]))
    #else:
    popt, pcov = curve_fit(func, np.array(Ns_meanss)[:,j], np.array(sigmas_s)[:,j]/70, sigma=np.array(sigmas_unc_s)[:,j]/70)
    alphas.append(popt[0])
    alphas_std.append(np.sqrt(pcov[0,0]))

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ax.scatter(np.array(investigated_values)/(8*625**3)*10**6, np.array(alphas), marker='^', s=100, c='r', zorder=3)
c[3] = 0.7
ax.plot(np.array(investigated_values)/(8*625**3)*10**6, np.array(alphas), c='r', zorder=2)
ax.errorbar(np.array(investigated_values)/(8*625**3)*10**6, np.array(alphas), yerr=np.array(alphas_std), capsize=5, c='r', fmt='None', zorder=1)

ax.grid(ls='dashed', c='cadetblue', alpha=0.7, zorder=0)
#ax.set_xlim(0.035,0.3)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
#ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 15)

#ax.legend(title=r'$\bar{N}$', title_fontsize=30, fontsize = 27, framealpha=1, loc='upper left', handletextpad=0.01, columnspacing=0.01, ncol=len(Ns))
ax.set_ylabel(r'$\alpha$', fontsize=35, labelpad=15)
#ax.set_xlabel(r'$N_{\mathrm{gal}}$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$n_*\,\times 10^{-6}$ Mpc$^{-3}$', fontsize=35, labelpad=15)

#ax.set_ylim(-0.01,0.25)
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


#%%

n_dens = np.array(investigated_values)/(8*625**3)

#%%






























































# %%


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

sigmas_s_1 = sigmas_s.copy()
sigmas_s_1[-2] = sigmas_s_1[-2]+[None]
sigmas_s_1[-1] = sigmas_s_1[-1]+[None]
sigmas_unc_s_1 = sigmas_unc_s.copy()
sigmas_unc_s_1[-2] = sigmas_unc_s_1[-2]+[None]
sigmas_unc_s_1[-1] = sigmas_unc_s_1[-1]+[None]

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)):
    c = next(color)
    edgecolor='white'
    if investigated_values[j] == 16000:
        #c=np.array([0,0,1,1])
        ax.scatter(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, c=c, zorder=2)
        ax.errorbar(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, yerr=np.array(sigmas_unc_s_1)[:-2,j]/70, capsize=5, c=c, fmt='None', zorder=1)
        c[3] = 1
        ax.scatter(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        
    else:
        ax.scatter(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, c=c, zorder=2)
        ax.errorbar(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, yerr=np.array(sigmas_unc_s_1)[:,j]/70, capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 20, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}/H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar N$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()

#%%


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

color = iter(cm.winter_r(np.linspace(0, 1, len(investigated_values))))
for j in range(len(investigated_values)):
    c = next(color)
    edgecolor='white'
    if investigated_values[j] == 16000:
        #c=np.array([0,0,1,1])
        ax.scatter(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, c=c, zorder=2)
        ax.errorbar(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, yerr=np.array(sigmas_unc_s_1)[:-2,j]/70, capsize=5, c=c, fmt='None', zorder=1)
        c[3] = 1
        ax.scatter(np.array(Ns)[:-2], np.array(sigmas_s_1)[:-2,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        
    else:
        ax.scatter(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, marker='^', s=100, c=c, label=r'$\#$ Gal $= {}$'.format(investigated_values[j]), zorder=3)
        c[3] = 0.7
        ax.plot(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, c=c, zorder=2)
        ax.errorbar(np.array(Ns), np.array(sigmas_s_1)[:,j]/70, yerr=np.array(sigmas_unc_s_1)[:,j]/70, capsize=5, c=c, fmt='None', zorder=1)

ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
#ax.set_xlim(50,100)
#ax.set_ylim(0,ymax)
#ax.grid(axis='both', ls='dashed', alpha=0.5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 20, framealpha=1)
ax.set_ylabel(r'$\sigma_{H_0}/H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
ax.set_xlabel(r'$\bar N$', fontsize=35, labelpad=15)
#ax.set_ylim(-0.01,0.2)
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()


# %%
