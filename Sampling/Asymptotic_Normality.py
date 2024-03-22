#matplotlib.rcParams['font.family'] = 'Computer Modern Serif'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

title = 'Posterior Asymptotic Normality & Constraining Power'

investigated_characteristic = 'event_num_log_powerpoint'
investigated_values = [2**n for n in range(1,9)]
max_numbers = ["0"]*len(investigated_values)

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
plot_lim_min = 4
rate = N[-1]/N[-2]
plot_lim_max = rate*N[-1] + 1

# ax.set_ylim(outer_x_min,outer_x_max)
# ax.set_xlim(plot_lim_min, plot_lim_max)

mus = []
sigs = []
maxs = []
argmaxs = []

space = N.tolist()
space.append(N[-1]*2)

axs = []


sigs = [np.mean(stds) for stds in stdss]
yerr = [np.std(stds) for stds in stdss]

def asymptot(N,a):
    return a/np.sqrt(N)

Ns = np.linspace(0.9*min(N), max(N)*1.1, 100)

popt, pcov = curve_fit(asymptot, N, sigs, sigma = yerr)
plt.plot(Ns, asymptot(Ns, popt[0]), '--r', label = r'$\frac{\alpha}{\sqrt{N}}$')

ax.errorbar(N, sigs, yerr = yerr, ls="", capsize = 5, color = 'dodgerblue', marker = 'd', label = r"$\langle{\hat{\sigma}_{H_0}\rangle}$", markerfacecolor= 'k', markeredgecolor= 'k')


ax.set_xscale('log')
N_labels = [str(i) for i in N]
ax.set_xticks(N)
ax.set_xticklabels(N_labels)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax.grid(axis='y', ls='dashed', alpha=0.7)
ax.set_xlabel('Average number of detected mergers', fontsize=35, labelpad=15)
ax.set_ylabel(r'$\hat{\sigma}_{H_0}$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
#ax.set_title(title, x=0.46, fontsize=35, pad=30)
#ax.legend(fontsize=28, framealpha=1, loc=(0.357,0.705))
ax.legend(fontsize=28, framealpha=1, loc = 'upper right')
ax.set_yscale("log")
ax.yaxis.grid(True, which='minor', ls = '--', alpha = 0.25)
# ax.set_xscale("log")

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//asymptotic_normality.svg'

plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()
