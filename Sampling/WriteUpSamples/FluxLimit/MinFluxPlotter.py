import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#SampleUniverse_3_50_0.1_50_3508.csv


# spec = gridspec.GridSpec(ncols=1, nrows=3,
#                          height_ratios=[4, 1, 1], wspace=0.2,
#                          hspace=0.2)

fig, ax = plt.subplots(2,1, layout="constrained", figsize = (10,8))
sim_type = ["Luminosity Proportional Galaxy Selection","Random Galaxy Selection", ]
color = ['blue', 'green']
files = ['SampleUniverse_3_6.366197723675814e-06_1_1000_0.00148777872778_1.csv','SampleUniverse_3_6.366197723675814e-06_1_1000_0.00148777872778_0.csv']

for i, file in enumerate(files):
    df = pd.read_csv(file, index_col=0)


    true_means = np.empty((len(df.columns)//7))
    means= []
    for _ in range(6):
        means.append([])
    # means = np.empty((len(df.columns)//7, 6))
    pct_limits = []
    for j, column in enumerate(df.columns):
        if j//7 == 0 and j!=0:
            pct_limits.append(np.float32(re.split("_", column)[2]))

        # if df[column].sum() ==0:
        pdf_single = df[column]/df[column].sum()
        mean = np.sum(pdf_single * df.index)
        if j%7 == 0:
            true_means[j//7] = mean
        elif mean!=0:
            means[j % 7 - 1].append(mean - true_means[j//7])


    # means -= np.min(means)*1.01
    # mean_means = np.mean(means, axis = 0)
#
# print(np.min(true_means))
    xlabels = [str(pct) for pct in pct_limits]
    xlabels.insert(0, None)
    violin = ax[i].violinplot(means, showmeans=True)
    ax[i].set_xticklabels(xlabels)
    if i == 1:
        ax[i].set_xlabel(r"$\frac{F_{min}}{F^{*}}$", fontsize = 14)
    ax[i].set_ylabel(r"$\Delta\,{\mu}$")
# fig.constrain_layout()
#     ax[i].set_title(sim_type[i])

    for pc in violin["bodies"]:
        pc.set_facecolor(color[i])
        pc.set_edgecolor(color[i])
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = violin[partname]
        vp.set_edgecolor(color[i])
        vp.set_linewidth(1)
    # if i == 0:
    ax[i].legend(*zip(*[(mpatches.Patch(color=color[i], alpha  = 0.3), sim_type[i])]), loc = 3)

plt.show()

    #     means.append(mean)
    #     stds.append(np.sqrt(np.sum((pdf_single * df.index ** 2)) - mean ** 2))
    # mean_df[str(nums[i])] = means
    # std_df[str(nums[i])] = stds
    # mean_of_means.append(np.mean(means))
    # mean_of_stds.append(np.mean(stds))
# mean_df = mean_df.reindex(sorted(mean_df.columns), axis=1)
# x_labels = (total_lum*rate*np.array(times)).astype(int)

