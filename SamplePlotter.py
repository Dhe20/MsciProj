#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections

df = pd.read_csv("SampleUniverse_3_50_0.1_50.csv", index_col = 0)


spec = gridspec.GridSpec(ncols=1, nrows=3,
                         height_ratios=[4, 1, 1], wspace=0.2,
                         hspace=0.2)



fig = plt.figure()

# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(8)

# set width of each subplot as 8
fig.set_figwidth(16)

# create grid for different subplots
spec = gridspec.GridSpec(ncols=3, nrows=1,
                        wspace=0.2,
                         hspace=0.2)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])

means = []
stds = []

df_new = df.reset_index()
for i in range(len(df.columns)-1):
    column = str(i)
    ax1.plot(df[column])
    mean = (df_new[column]*df_new['index']).sum()
    means.append(mean)
    std = np.sqrt((df_new[column]*((mean - df_new['index'])**2)).sum())
    stds.append(std)
ax2.hist(means)
ax3.hist(stds)
plt.show()

# %%
