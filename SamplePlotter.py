import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections

df = pd.read_csv("SampleUniverse_2_20_0.1_50.csv", index_col = 0)

fig = plt.figure()
plt.subplots()
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
for column in df.columns:
    ax1.plot(df[column])
    means.append(df[column].mean())
    stds.append(df[column].std())
ax2.hist(means)
ax3.hist(stds)
plt.show()
