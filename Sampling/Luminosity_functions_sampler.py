#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import gridspec, collections
from Sampling.ClassSamples import Sampler
from tqdm import tqdm

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

#%%

investigated_characteristic = 'survey_type'
investigated_values = ['perfect', 'imperfect']
max_numbers = []
b = []
f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(survey_type= investigated_values[i], universe_count = 10, resolution_H_0=75,
                            total_luminosity=200/3, sample_time=0.0135, redshift_noise_sigma=500,
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    b.append(Investigation.burr_i)
    f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
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
'''
investigated_characteristic = 'characteristic_luminosity'
investigated_values = list(np.array([0.01,0.05,0.1,0.5,1]))
max_numbers = []
# Temporarily
max_numbers = ['0','0','0','0','0']


for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(characteristic_luminosity=investigated_values[i], total_luminosity = 100, 
                            sample_time = 0.009947, universe_count = 50, investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)

'''

#%%

investigated_characteristic = 'total_luminosity'
investigated_values = list(np.array([10,100,1000,10000]))
max_numbers = []

for i in tqdm(range(len(investigated_values))):
    time = 50/(f*1540*investigated_values[i])
    Investigation = Sampler(total_luminosity = investigated_values[i],
                            sample_time = time, universe_count = 50, 
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'BVM_kappa'
investigated_values = list(np.array([20,50,100,200,500]))
max_numbers = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(BVM_kappa = investigated_values[i], universe_count = 50, 
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'BVM_c'
investigated_values = list(np.array([5,10,50,100,200]))
max_numbers = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(BVM_c = investigated_values[i], universe_count = 50, 
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'BVM_k'
investigated_values = list(np.array([0.5,1,2,5,20]))
max_numbers = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(BVM_k = investigated_values[i], universe_count = 50, 
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'redshift_noise_sigma_corrected'
investigated_values = [10,100,1000,2500]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(survey_type='imperfect', universe_count = 20 ,redshift_noise_sigma=investigated_values[i],
                            total_luminosity=200/3, sample_time=0.014, resolution_H_0=100, investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_noise_sigma'
investigated_values = [100]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 50,redshift_noise_sigma=investigated_values[i],
                            investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'noise_distribution'
investigated_values = ['gauss', 'gauss', 'BVMF_eff', 'BVMF_eff']
infer_gauss = [True, False, True, False]
titles = ['gauss_gauss', 'BVM_gauss', 'gauss_BVMF_eff', 'BVM_BVMF_eff']
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, total_luminosity=1000/3, sample_time = 0.00242, gauss=infer_gauss[i], noise_sigma=10,
                            noise_distribution = investigated_values[i], investigated_characteristic = investigated_characteristic,
                            investigated_value = titles[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

# new c
# new cube
investigated_characteristic = 'cube'
investigated_values = [True, False]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, cube = investigated_characteristic[i], total_luminosity=1000/3, sample_time = 0.00242, 
                            noise_distribution = 'BVMF_r2_eff', investigated_characteristic = investigated_characteristic,
                            investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'gauss_random_p_det'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 50, p_det=investigated_values[i], gamma = False, gauss=True, event_distribution='Random', total_luminosity=1000/3, sample_time = 0.00242,
                            noise_distribution='gauss', event_distribution_inf='Random', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'bvmf_random_p_det'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 30, p_det=investigated_values[i], gamma = False, event_distribution='Random', total_luminosity=100/3, sample_time = 0.01162,
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'bvmf_random_p_det_imperfect'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 10, p_det=investigated_values[i], gamma = False, survey_type = "imperfect", event_distribution='Random', total_luminosity=100/3, sample_time = 0.01162, resolution_H_0 = 100,
                           redshift_noise_sigma=0.05, noise_distribution='BVMF_eff', event_distribution_inf='Random', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'bvmf_random_p_det_imperfect_notinf'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 10, p_det=investigated_values[i], gamma = False, survey_type = "perfect", event_distribution='Random', total_luminosity=100/3, sample_time = 0.01162, resolution_H_0 = 100,
                           redshift_noise_sigma=0.05, noise_distribution='BVMF_eff', event_distribution_inf='Random', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

#investigated_values = list(np.array([5,10,50,100,200]))
max_numbers = ['0','0', '0', '0', '0']
#investigated_values = list(np.array([10,100,1000]))

investigated_characteristic = 'BVM_c'
investigated_values = list(np.array([5,10,50,100,200]))

investigated_characteristic = 'luminosity_gen_type'
investigated_values = ['Fixed', 'Full-Schechter']
max_numbers = ['0','0']

#%%

def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

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
    #print(i)
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    means = []
    stds = []
    for column in df.columns:
        pdf_single = df[column]/df[column].sum()
        pdf_single.dropna(inplace=True)
        mean = sum(pdf_single*pdf_single.index)
        if mean==0:
            continue
        means.append(mean)
        stds.append(np.sqrt(sum((pdf_single*pdf_single.index**2))-mean**2))
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)

ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

ax1.violinplot(meanss, vert=False, showmeans=True)
ax1.set_yticks(pos)
ax1.set_yticklabels(investigated_values, fontsize=20)
ax1.set_title('Means', fontsize = 25)
ax1.grid(axis='x')

ax2.violinplot(stdss, vert=False, showmeans=True)
ax2.set_yticks(pos)
ax2.set_yticklabels(investigated_values, fontsize=20)
ax2.set_title('Standard deviations', fontsize = 25)
fig.supylabel(axis_namer(investigated_characteristic), fontsize=20)
ax2.grid(axis='x')

plt.show()






# %%
