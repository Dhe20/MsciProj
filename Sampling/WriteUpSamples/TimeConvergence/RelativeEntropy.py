#%%
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

# x = [4.5, 19.7, 38.6]
mean_of_means = []
mean_of_stds = []

import os
path = os.getcwd()
csv_list = []
name = ".csv"
for file in os.listdir(path):
    # if dirs != []:
    #     continue
    #     # dir = dirs[0]
    # else:
    #     for file in files:
    if name in file:
        csv_list.append(file)
nums = []
times = []
for csv in csv_list:
    nums.append(float(csv.split("_", -1)[-2].split("_")[0]))
    times.append(float(csv.split("_", -1)[-3].split("_")[0]))
total_lum = float(csv_list[0].split("_", -1)[2].split("_")[0])
rate = float(csv_list[0].split("_", -1)[4].split("_")[0])

for i, file in enumerate(csv_list):
    df = pd.read_csv(file, index_col=0)
    number_of_universes = len(df.columns)
    if i == 0:
        for j, column in enumerate(df.columns):
            if j == 0:
                pdf_single = df[column] / df[column].sum()
                H_0_resolution = len(pdf_single)

number_of_time_samples = len(csv_list)
posterior_converging = np.zeros((number_of_universes, number_of_time_samples, H_0_resolution))

times, nums, csv_list  = zip(*sorted(zip(times, nums, csv_list)))

import math
from scipy.stats import entropy as H

def JSD(prob_distributions, weights = 0, logbase=2):
    # left term: entropy of mixture
    if weights == 0:
        weights = 1/len(prob_distributions)
    wprobs = prob_distributions*weights
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    # right term: sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_distributions])
    wentropies = weights * entropies
    # wentropies = np.dot(weights, entropies)
    sum_of_entropies = wentropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return(divergence)
Mixture = []
WentropySum = []
js_distance_arr = np.zeros(number_of_time_samples)
for i, file in enumerate(csv_list):
    df = pd.read_csv(file, index_col=0)
    pdf_arr = []
    pdf_arr_x = []
    for j, column in enumerate(df.columns):
        if df[column].sum() > 0.0:
            pdf_arr.append(df[column].values/df[column].sum())
    if i == 0:
        # print(pdf_arr)
        x = np.sum(np.log(pdf_arr), axis = 0)
        x -= np.max(x)
        x = np.exp(x)
        plt.plot(df.index, x)
        # plt.plot(df.index, pdf_arr[100])
        plt.show()
    # js_distance_arr[i] = JSD(np.array(pdf_arr))
# print(js_distance_arr)

# plt.plot(js_distance_arr, '-x')
# plt.plot(WentropySum, '-x')
# plt.plot(Mixture, 'x')





# # js_distance_arr = js_distance_arr[np.logical_not(np.isnan(js_distance_arr))]
# # idxmax = np.argmax(np.mean(js_distance_arr, axis = 1))
# # idxmin = np.argmin(np.mean(js_distance_arr, axis = 1))
#
# # for i, _ in enumerate(js_distance_arr):
# #     plt.plot(nums[:-1], js_distance_arr[i], "x")
# #
# # plt.errorbar(nums[:-1], np.mean(js_distance_arr, axis = 0), yerr = np.std(js_distance_arr, axis = 0), fmt= "k", capsize = 2)
# # # plt.fill_between(nums[:-1], np.min(js_distance_arr, axis = 0), np.max(js_distance_arr, axis = 0), alpha  = 0.2, color = "k")
# # plt.fill_between(nums[:-1], js_distance_arr[idxmin], js_distance_arr[idxmax] , alpha  = 0.2, color = "k")
# # plt.show()


# js_distance_arr = js_distance_arr[np.logical_not(np.isnan(js_distance_arr).any(axis=1)), :]
# df = pd.DataFrame(js_distance_arr, columns = [str(num) for num in nums[:-1]])
# df2 = df.stack().reset_index()
# df2.columns = ['Universe_Number','AvgEventNum','JSDist']
# df2['AvgEventNum'] = pd.Categorical(df2.AvgEventNum, categories=pd.unique(df2.AvgEventNum))

# %%
#
# axis_label_font = 14
#
# x_labels = (total_lum*rate*np.array(times)).astype(int)
#
# data = [list(col) for col in js_distance_arr.transpose()]
#
# fig, ax1 = plt.subplots(figsize=(10, 6), layout="constrained")
# # fig.canvas.manager.set_window_title('A Boxplot Example')
# fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
#
# bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
# plt.setp(bp['medians'], color='black')
# plt.setp(bp['boxes'], color='black')
# plt.setp(bp['whiskers'], color='black')
# plt.setp(bp['fliers'], color='black', marker='+')
#
# # Add a horizontal grid to the plot, but make it very light in color
# # so we can use it for reading data values but not be distracting
# # ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
# #                alpha=0.5)
#
# ax1.set(
#     axisbelow=True,  # Hide the grid behind plot objects
# )
#
# # Now fill the boxes with desired colors
# box_colors = ['royalblue', 'royalblue']
# num_boxes = number_of_time_samples-1
# medians = np.empty(num_boxes)
# for i in range(num_boxes):
#     box = bp['boxes'][i]
#     box_x = []
#     box_y = []
#     for j in range(5):
#         box_x.append(box.get_xdata()[j])
#         box_y.append(box.get_ydata()[j])
#     box_coords = np.column_stack([box_x, box_y])
#     # Alternate between Dark Khaki and Royal Blue
#     ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
#     # Now draw the median lines back over what we just filled in
#     med = bp['medians'][i]
#     median_x = []
#     median_y = []
#     for j in range(2):
#         median_x.append(med.get_xdata()[j])
#         median_y.append(med.get_ydata()[j])
#         ax1.plot(median_x, median_y, 'k')
#     medians[i] = median_y[0]
#     # Finally, overplot the sample averages, with horizontal alignment
#     # in the center of each box
#     # ax1.plot(np.average(med.get_xdata()), np.mean(data[i]),
#     #          color='w', marker='*', markeredgecolor='k')
#
# # Set the axes ranges and axes labels
# ax1.set_xlim(0.5, num_boxes + 0.5)
# # top = np.max(js_distance_arr)*5
# # bottom = np.min(js_distance_arr)/2
# # ax1.set_ylim(bottom, top)
# ax1.set_xticklabels(x_labels[:-1], rotation=90, fontsize = 12)
#
#
# pos = np.arange(num_boxes) + 1
# upper_labels = nums[:-1]
# weights = ['bold', 'bold']
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
# ax2.set_xticklabels(nums[:-1], fontsize = 12, rotation=0)
# ax2.set_xlabel("Average Number of Events Detected", fontsize = axis_label_font)
# ax1.set_xlabel(r'$\Gamma{}TL_{0}$', fontsize = axis_label_font)
# ax1.set_ylabel('Relative Entropy', fontsize = axis_label_font)
# ax1.set_yticklabels(ax1.get_yticks(), fontsize = 12)
# plt.show()