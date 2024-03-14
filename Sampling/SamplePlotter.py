#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, collections

#%%
#SampleUniverse_3_50_0.1_50_3508.csv
df = pd.read_csv("PosteriorData\\SampleUniverse_trial_survey_completeness_2.5464790894703255e-07_0.csv", index_col = 0)
df = pd.read_csv("PosteriorData\\SampleUniverse_redshift_uncertainty_corrected_approx_0.005_0.csv", index_col = 0)
df = pd.read_csv("PosteriorData\\SampleUniverse_redshift_uncertainty_corrected_approx_0.005_0.csv", index_col = 0)

df.dropna(inplace=True, axis=1)

spec = gridspec.GridSpec(ncols=1, nrows=3,
                         height_ratios=[4, 1, 1], wspace=0.2,
                         hspace=0.2)

fig = plt.figure(figsize = (12,8))


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
    pdf_single = df[column]/df[column].sum()
    #ax1.plot(pdf_single)
    mean = sum(pdf_single*df.index)
    means.append(mean)
    stds.append(np.sqrt(sum((pdf_single*df.index**2))-mean**2))
ax2.hist(means, bins = 15, histtype='step', edgecolor='b', facecolor='lightblue', hatch='/', fill=True)
ax2.vlines(x=np.mean(means), ymin=0, ymax=13, label='Mean={:.2f}'.format(np.mean(means)))
ax3.hist(stds, bins = 15, histtype='step', edgecolor='b', facecolor='lightblue', hatch='/',  fill=True)
ax2.legend(fontsize=15, loc=8)
plt.show()
print(np.mean(means))
print(np.std(means))
print(np.mean(stds))

# %%
