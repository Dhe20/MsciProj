#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections
from plotnine import *

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
for csv in csv_list:
    nums.append(float(csv.split("_", -1)[-2].split("_")[0]))

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
std_df = std_df.reindex(sorted(std_df.columns), axis=1)
# print(mean_of_means)
# plt.plot(nums, mean_of_stds, ".")
# plt.legend()
# plt.show()

df2 = mean_df.stack().reset_index()
df2.columns = ['SampleNum','AvgEventNum','PosteriorMean']
df2['AvgEventNum'] = df2['AvgEventNum'].astype(float)
df3 = df2.sort_values(by = ["AvgEventNum", "SampleNum"]).reset_index()
# print(df3.AvgEventNum)
# g = (
# ggplot(df3, aes(x = 'AvgEventNum', y = 'PosteriorMean', group = 'AvgEventNum')) +
#   geom_boxplot()
# )
# print(g)

df2 = std_df.stack().reset_index()
df2.columns = ['SampleNum','AvgEventNum','PosteriorStd']
df2['AvgEventNum'] = df2['AvgEventNum'].astype(float)
df3 = df2.sort_values(by = ["AvgEventNum", "SampleNum"]).reset_index()

print(df3.head(10))

g = (
ggplot(df3, aes(x = 'AvgEventNum', y = 'PosteriorStd', group = 'AvgEventNum')) +
  geom_boxplot() + scale_y_continuous(trans='log10')
)
print(g)

# %%
