#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections
from plotnine import *
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections
# from plotnine import *
from scipy.spatial import distance
from plotnine.scales import scale_y_continuous
import scipy
from matplotlib.patches import Polygon

#SampleUniverse_3_50_0.1_50_3508.csv


# spec = gridspec.GridSpec(ncols=1, nrows=3,
#                          height_ratios=[4, 1, 1], wspace=0.2,
#                          hspace=0.2)

x = [4.5, 19.7, 38.6]
mean_of_means = []
mean_of_stds = []

import os
path = os.getcwd()
csv_list = []
name = ".csv"
for root, dirs, files in os.walk(path):
    if dirs == []:
        continue
    for file in files:
        if name in file:
            csv_list.append(file)
nums = []
times = []
for csv in csv_list:
    nums.append(float(csv.split("_", -1)[-2].split("_")[0]))
    times.append(float(csv.split("_", -1)[-3].split("_")[0]))
total_lum = float(csv_list[0].split("_", -1)[2].split("_")[0])
rate = float(csv_list[0].split("_", -1)[4].split("_")[0])

mean_df = pd.DataFrame()
std_df = pd.DataFrame()
for i, file in enumerate(csv_list):
    means = []
    stds = []
    df = pd.read_csv(file, index_col=0)
    for column in df.columns:
        # if df[column].sum() ==0:
        pdf_single = df[column]/df[column].sum()
        # plt.plot(pdf_single, label = nums[i])
        mean = np.sum(pdf_single*df.index)
        means.append(mean)
        stds.append(np.sqrt(np.sum((pdf_single * df.index ** 2)) - mean ** 2))
    mean_df[str(nums[i])] = means
    std_df[str(nums[i])] = stds
    mean_of_means.append(np.mean(means))
    mean_of_stds.append(np.mean(stds))
mean_df = mean_df.reindex(sorted(mean_df.columns), axis=1)
x_labels = (total_lum*rate*np.array(times)).astype(int)

# std_df = std_df.reindex([str(col for col in sorted([np.float32(col) for col in std_df.columns])), axis=1)
number_of_time_samples = len(csv_list)
# print(mean_of_means)
# plt.plot(nums, mean_of_stds, ".")
# plt.legend()
# plt.show()
#
# df2 = mean_df.stack().reset_index()
# df2.columns = ['SampleNum','AvgEventNum','PosteriorMean']
# df2['AvgEventNum'] = df2['AvgEventNum'].astype(float)
# df3 = df2.sort_values(by = ["AvgEventNum", "SampleNum"]).reset_index()
# # print(df3.AvgEventNum)
# # g = (
# # ggplot(df3, aes(x = 'AvgEventNum', y = 'PosteriorMean', group = 'AvgEventNum')) +
# #   geom_boxplot()
# # )
# # print(g)
#
# df2 = std_df.stack().reset_index()
# df2.columns = ['SampleNum','AvgEventNum','PosteriorStd']
# df2['AvgEventNum'] = df2['AvgEventNum'].astype(float)
# df3 = df2.sort_values(by = ["AvgEventNum", "SampleNum"]).reset_index()
#
# print(df3.head(10))
#
# g = (
# ggplot(df3, aes(x = 'AvgEventNum', y = 'PosteriorStd', group = 'AvgEventNum')) +
#   geom_boxplot() + scale_y_continuous(trans='log10')
# )
# print(g)



# %%
from scipy.optimize import curve_fit

from scipy.optimize import curve_fit



def poly(x, a):
    return a*x**-0.5

data = [list(std_df[col]) for col in std_df.columns]
# x_labels, data, nums = zip(*sorted(zip(x_labels, data, nums)))
nums, x_labels, data = zip(*sorted(zip(nums, x_labels, data)))

ys = np.array(data).flatten()
xs = np.repeat(np.array(x_labels), repeats = len(data[0]))
long_nums = np.repeat(np.array(nums), repeats = len(data[0]))
long_nums = long_nums[ys != -np.inf]
xs = xs[ys != -np.inf]
ys = ys[ys !=-np.inf]

asymptotic_regime = 30
ys = ys[np.array(long_nums) > asymptotic_regime]
xs = xs[np.array(long_nums) > asymptotic_regime]

mean = np.mean(data, axis = 1)
std =  np.std(data, axis = 1)

fit, _ = curve_fit(f = poly, xdata= xs, ydata = ys)
# plt.xlim(min(xs)-0.1, max(xs)+0.1)

min_val = 0
min_y_lim = 0.5

xfine = np.linspace(np.min(x_labels), np.max(x_labels), 1000)

#
# fig, ax1 = plt.subplots(figsize=(10, 6), layout="constrained")
#
# axis_label_font = 14
#
# plt.plot(x_labels, mean, '+', c = "k")
# plt.plot(xfine, poly(xfine, a = fit), 'k--')
# # print(np.max([mean - std,np.full((len(mean)), 0)], axis = 0))
# plt.fill_between(x_labels, np.max([mean - std,np.full((len(mean)), min_val)], axis = 0), mean + std, color = 'green', alpha = 0.2, edgecolor = None)
# plt.fill_between(x_labels, np.max([mean - 2*std,np.full((len(mean)), min_val)], axis = 0), mean + 2*std, color = 'green', alpha = 0.2, edgecolor = None)
# plt.fill_between(x_labels, np.max([mean - 3*std,np.full((len(mean)), min_val)], axis = 0), mean + 3*std, color = 'green', alpha = 0.2, edgecolor = None)
# plt.yscale("log")
# # plt.xscale("log")
# plt.ylim(min_y_lim, np.max([mean + 3*std,np.full((len(mean)), min_val)]))
#
#
# # ax1.set_xticklabels(x_labels[:], rotation=90, fontsize = 12)
# # weights = ['bold', 'bold']
# # for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
# #     k = tick % 2
# #     ax1.text(pos[tick], .95, upper_labels[tick],
# #              transform=ax1.get_xaxis_transform(),
# #              horizontalalignment='center', size='small',
# #              weight=weights[k], color="k")
#
# ax1.set_yscale("log")
# ax2 = ax1.twiny()
# ax2.set_navigate(False)
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(ax1.get_xticks())
# ax2.set_xticklabels(np.round(nums[:],1), fontsize = 12, rotation=0)
# ax2.set_xlabel("Average Number of Events Detected", fontsize = axis_label_font)
# ax1.set_xlabel(r'$\Gamma{}TL_{0}$', fontsize = axis_label_font)
# ax1.set_ylabel('Posterior Standard Deviation', fontsize = axis_label_font)
#
# plt.show()





# %%
# #

import matplotlib.ticker as ticker


axis_label_font = 14


fig, ax1 = plt.subplots(figsize=(10, 6), layout="constrained")
# fig.canvas.manager.set_window_title('A Boxplot Example')
# fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
plt.setp(bp['medians'], color='black')
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='black', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
# ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                alpha=0.5)

ax1.set(
    axisbelow=True,  # Hide the grid behind plot objects
)

# Now fill the boxes with desired colors
box_colors = ['white', 'white']
num_boxes = number_of_time_samples
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')
    medians[i] = median_y[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    # ax1.plot(np.average(med.get_xdata()), np.mean(data[i]),
    #          color='w', marker='*', markeredgecolor='k')

ax1.plot([i for i in range(1,len(x_labels)+1)] , poly(np.array(x_labels).astype(np.float32), a = fit), ls = '--', c = "red", label = "Asymptotic Relation")
ax1.plot(1, 1,  'k+', label = "Outliers")

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, num_boxes + 0.5)
# top = np.max(js_distance_arr)*5
# bottom = np.min(js_distance_arr)/2
# ax1.set_ylim(bottom, top)
ax1.set_xticklabels(x_labels[:], rotation=90, fontsize = 12)


pos = np.arange(num_boxes) + 1
upper_labels = nums[:]
weights = ['bold', 'bold']
# for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
#     k = tick % 2
#     ax1.text(pos[tick], .95, upper_labels[tick],
#              transform=ax1.get_xaxis_transform(),
#              horizontalalignment='center', size='small',
#              weight=weights[k], color="k")

ax1.set_yscale("log")
ax2 = ax1.twiny()
# ax2.set_navigate(False)
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(np.round(nums[:],1), fontsize = 12, rotation=0)
ax2.set_xlabel("Average Number of Events Detected", fontsize = axis_label_font)
ax1.set_xlabel(r'$\Gamma{}TL_{0}$', fontsize = axis_label_font)
ax1.set_ylabel('Posterior Standard Deviation', fontsize = axis_label_font)
# ax1.set_yticklabels(ax1.get_yticks(), fontsize = 12)
# ax1.set_yticklabels([0.5, 1, 5, 10])
start, end = ax1.get_ylim()
ax1.yaxis.set_ticks([0.5, 1, 5, 10])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax1.legend()
plt.show()

#%%

