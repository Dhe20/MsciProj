
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import gridspec, collections
from ClassSamples import Sampler
from tqdm import tqdm

'''

investigated_characteristic = 'luminosity_gen_type'
investigated_values = ['Fixed', 'Full-Schechter']
max_numbers = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(luminosity_gen_type = investigated_values[i], total_luminosity = 100, 
                            sample_time = 0.009947, universe_count = 200, investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)

''' 
investigated_characteristic = 'characteristic_luminosity'
investigated_values = list(np.array([0.01,0.05,0.1,0.5,1]))
max_numbers = []

'''
for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(characteristic_luminosity=investigated_values[i], total_luminosity = 100, 
                            sample_time = 0.009947, universe_count = 50, investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)

'''

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
for i in range(len(investigated_values)):
    #SampleUniverse_3_50_0.1_50_3508.csv
    filename = "SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    means = []
    stds = []
    for column in df.columns:
        pdf_single = df[column]/df[column].sum()
        mean = sum(pdf_single*df.index)
        means.append(mean)
        stds.append(np.sqrt(sum((pdf_single*df.index**2))-mean**2))
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)

ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

ax1.violinplot(meanss, vert=False)
ax1.set_yticks(pos)
ax1.set_yticklabels(investigated_values, fontsize=20)
ax1.set_title('Means', fontsize = 25)
ax2.violinplot(stdss, vert=False)
ax2.set_yticks(pos)
ax2.set_yticklabels(investigated_values, fontsize=20)
ax2.set_title('Standard deviations', fontsize = 25)
fig.supylabel(investigated_characteristic)
plt.show()



