#%%

import pandas as pd
import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

#%%

#def V_loc(kappa, s_D_D, D_max):
#    s_kappa = np.sqrt(1-iv(1,kappa)/iv(0,kappa))
#    return ((16*3*2)/(7*np.pi))*s_kappa*s_kappa*s_D_D*D_max**3

def V_loc(kappa, s_D_D, D_max):
    s_kappa = np.sqrt(1-iv(1,kappa)/iv(0,kappa))
    return ((16*8)/(np.pi))*s_kappa*s_kappa*s_D_D*D_max**3

def V_centroid(s_g, S):
    return (4*np.pi/3)*(s_g*S)**3

#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

title = 'Centroid'

#investigated_characteristic = "CentroidSigma"
#investigated_characteristic = "CentroidSigma_sigma_D_30%"
N_centroids = [10,15,20,25]
# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2]
investigated_values = [0.04, 0.08]
# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]
# max_numbers = ["0","0","0","0","0", "2", "0", "0", "0", "0", "2"]

investigated_characteristic = "20LocVol_CentroidSigma"
investigated_values = [0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.36, 0.48, 0.64, 1.0, 2.0]
N_centroids = [10, 15, 20, 25]

max_numbers = ["0"]*len(investigated_values)
event_count_max_numbers =  ["0"]*len(investigated_values)

spectral_map = cm.get_cmap('winter', len(N_centroids))
colors = spectral_map(np.linspace(0, 1, len(N_centroids)))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,12))
fig.subplots_adjust(hspace=0)

scale = 'vol'

for j, N_gal in enumerate(N_centroids):
    means = []
    stds = []
    alphas = []
    # for column in df.columns:
    post_avg = []
    N = np.array(investigated_values)
    meanss = []
    stdss = []
    alphass = []
    pos = []
    df_N = pd.DataFrame()

    for i in range(len(investigated_values)):
        # print(i)
        filename = "SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                   max_numbers[i] + ".csv"
        df = pd.read_csv(filename, index_col=0)
        df.dropna(inplace=True, axis=1)
        filename = "EventCount_SampleUniverse_" + str(investigated_characteristic) + "_" + str(N_gal)+ "_" +str(investigated_values[i]) + "_" + \
                   event_count_max_numbers[i] + ".csv"
        df_event_count = pd.read_csv(filename, index_col=0)
        df_event_count.dropna(inplace=True, axis=1)
        means = []
        stds = []
        alphas = []
        for i, column in enumerate(df.columns):
            pdf_single = df[column] / df[column].sum()  # * (df.index[1] - df.index[0])
            # pdf_single.dropna(inplace=True)
            vals = np.array(pdf_single.index)
            event_count = df_event_count[column][0]
            mean = sum(pdf_single * vals)
            # means or modes
            # mean = vals[np.argmax(pdf_single*vals)]
            means.append(mean)
            stds.append(np.sqrt(sum((pdf_single * pdf_single.index ** 2)) - mean ** 2))
            alphas.append(np.sqrt(sum((pdf_single * pdf_single.index ** 2)) - mean ** 2) * np.sqrt(event_count))
        # df_N[str(investigated_values[i])] = df.mean(axis=1)
        # df_N[str(investigated_values[i])] = df_N[str(investigated_values[i])] / df_N[
        #     str(investigated_values[i])].sum()
        if j == 0:
            good_samples = list(np.where(df_event_count.values[0]>=10)[0])
        means = [mean for i, mean in enumerate(means) if i in good_samples]
        stds = [std for i, std in enumerate(stds) if i in good_samples]
        alphas = [alpha/70 for i, alpha in enumerate(alphas) if i in good_samples]
        # stdss = stdss[good_samples]
        # alphass = alphass[good_samples]

        meanss.append(means)
        stdss.append(stds)
        alphass.append(alphas)
        pos.append(i + 1)

    # print(len(meanss))

    stdss = [list(np.array(stds)/70) for stds in stdss]

    v_cent_loc = V_centroid(np.array(investigated_values),625)/V_loc(20,0.2,250)

    if scale == 'normal': 
        ax1.plot(investigated_values, [np.mean(alphas) for alphas in alphass] ,"--", lw=3, color = colors[j])
        ax2.plot(investigated_values, [np.mean(stds) for stds in stdss] ,"--", lw=3, color = colors[j])
        ax1.scatter(investigated_values, [np.mean(alphas) for alphas in alphass], marker='x', s=150, linewidths=2, color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))
        ax2.scatter(investigated_values, [np.mean(stds) for stds in stdss], marker='x', s=150, linewidths=2, color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))
        ax1.errorbar(investigated_values, [np.mean(alphas) for alphas in alphass], yerr=[np.std(alphas)/np.sqrt(len(alphas)) for alphas in alphass], color = colors[j], capsize=5)
        ax2.errorbar(investigated_values, [np.mean(stds) for stds in stdss], yerr=[np.std(stds)/np.sqrt(len(stds)) for stds in stdss], color = colors[j], capsize=5)
    elif scale == 'vol':
        ax1.plot(v_cent_loc, [np.mean(alphas) for alphas in alphass] ,"--", lw=3, color = colors[j])
        ax2.plot(v_cent_loc, [np.mean(stds) for stds in stdss] ,"--", lw=3, color = colors[j])
        ax1.scatter(v_cent_loc, [np.mean(alphas) for alphas in alphass], marker='x', s=150, linewidths=2, color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))
        ax2.scatter(v_cent_loc, [np.mean(stds) for stds in stdss], marker='x', s=150, linewidths=2, color = colors[j], label = r'$N_c = {}$'.format(str(N_gal)))
        ax1.errorbar(v_cent_loc, [np.mean(alphas) for alphas in alphass], yerr=[np.std(alphas)/np.sqrt(len(alphas)) for alphas in alphass], color = colors[j], capsize=5)
        ax2.errorbar(v_cent_loc, [np.mean(stds) for stds in stdss], yerr=[np.std(stds)/np.sqrt(len(stds)) for stds in stdss], color = colors[j], capsize=5)



ax1.set_ylabel(r"$\alpha_{\sigma_{H_0}/H_0}$", fontsize = 40, labelpad=10)

if scale == 'normal': 
    ax2.set_xlabel(r"$\sigma_g/S$", fontsize = 40, labelpad=10)
elif scale == 'vol':
    ax2.set_xlabel(r"$V_{\mathrm{centroid}}/V_{\mathrm{loc}}$", fontsize = 40, labelpad=10)
    ax2.set_xscale('log')
    ax1.set_xscale('log')

ax2.set_ylabel(r"$\sigma_{\hat{H}_0}/H_0$", fontsize = 40, labelpad=10)

ax1.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
ax1.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax1.legend(fontsize = 30, loc=8, framealpha=1)

ax2.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
#ax2.legend(fontsize = 30, loc=8, framealpha=1)


plt.show()
# %%
