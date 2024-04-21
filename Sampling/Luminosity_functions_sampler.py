#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sps
from scipy.optimize import curve_fit
import seaborn as sn
from matplotlib import gridspec, collections
from Sampling.ClassSamples import Sampler
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

#%%

def bias_dist(x, H=70):
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps

def C_I_samp(x):
    H = np.random.uniform(50,100)
    H = 70
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps



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
    
investigated_characteristic = 'bvmf_prop_p_det'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 50, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, sample_time = 0.001937,
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
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
    
investigated_characteristic = 'bvmf_proportional_p_det_many'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 2000, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'sel_eff_standard_01'
investigated_values = [False, True]
max_numbers = []
#b = [
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1000, p_det=investigated_values[i], beta=-1.3, lower_lim=0.1, gamma = False, event_distribution='Proportional', specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'selection_effects'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 50, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 100, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%


investigated_characteristic = 'selection_effects_standard'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1000, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'selection_effects_standard_2'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1000, beta=-1.5, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'selection_effects_standard_3'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1000, beta=-1.3, lower_lim=0.1, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)



#%%

# IGNORE
investigated_characteristic = 'YOYOYO'
investigated_values = [False, True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 50, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, characteristic_luminosity=100, wanted_det_events = 20, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty'
investigated_values = [0.0001,0.003,0.01]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 1000, redshift_noise_sigma=investigated_values[i] , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'redshift_uncertainty_standard_good'
investigated_values = [0.001,0.005,0.01]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 1000, redshift_noise_sigma=investigated_values[i] , p_det=True, gamma = False, event_distribution='Proportional', beta=-1.3,lower_lim=0.1, specify_gal_number=True, wanted_gal_n=5000, total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'redshift_uncertainty_corrected_approx'
investigated_values = [0.005]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 20, survey_type='imperfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=500, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'redshift_uncertainty_corrected_wwwrong'
investigated_values = [0.005]
max_numbers = []
#b = []
#f = []

wanted_gal_n = 2000
wanted_det_events = 10
d_ratio = 0.6
beta = -1.3


for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, beta=beta, d_ratio=d_ratio, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=wanted_gal_n, wanted_det_events = wanted_det_events, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    

investigated_characteristic = 'redshift_uncertainty_corrected_wwwrong_2'
investigated_values = [0.01]
#investigated_values = [0.005]
max_numbers = []
#b = []
#f = []

wanted_gal_n = 3000
wanted_det_events = 10
d_ratio = 0.7
beta = -1.3


for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 200, beta=beta, d_ratio=d_ratio, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=wanted_gal_n, wanted_det_events = wanted_det_events, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)



#%% 











#%%
    
investigated_characteristic = 'survey_incompleteness'
#investigated_values = [25,75,95]
investigated_values = np.array([0.01,0.1,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 200, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'survey_incompleteness_standard_good'
#investigated_values = [25,75,95]
#investigated_values = np.array([0.01,0.1,0.5])
investigated_values = np.array([0.3])
investigated_values /= (4*np.pi*(0.4*625)**2)
#max_numbers = []
#percentage = []
#b = []
#f = []
new_pecentage = percentage[:2]

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1000, min_flux=investigated_values[i], beta=-1.3, lower_lim=0.1, completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    new_pecentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'survey_incompleteness_many'
#investigated_values = [25,75,95]
investigated_values = np.array([0,1,2])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage1 = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 2000, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage1.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
for i in range(len(percentage1)):
    print(100*(1-np.mean(percentage1[i])))


#%%

investigated_characteristic = 'event_num'
investigated_values = [20, 40, 60, 80, 100]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = investigated_values[i], specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'event_num_log'
investigated_values = [5, 10, 20, 40, 80, 160]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = investigated_values[i], specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'single_event_data'
investigated_values = [True]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 1, p_det=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', poster=True, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

# RUN THIS ONE

investigated_characteristic = 'trial_survey_completeness_wrong_2'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 500, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Shoddy-Schechter' , lum_function_inf='Shoody-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
# Consistency check
investigated_characteristic = 'trial_survey_completeness_wrong_2_based_on_distance'
#investigated_values = [25,75,95]
lstars = np.array([0.05,0.1,0.2,0.3]) #,0.5])
#investigated_values /= (4*np.pi*(0.4*625)**2)
investigated_values = 0.4*625 * (lstars)**(-0.5)
investigated_values = np.array([500,400,300,200])
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 500, min_flux=0, completeness_type='cut_lim', DD = investigated_values[i], p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Shoddy-Schechter' , lum_function_inf='Shoody-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
# RUN THIS ONE

investigated_characteristic = 'trial_survey_completeness_wrong_3'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 500, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Full-Schechter' , lum_function_inf='Full-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
# RUN THIS ONE

investigated_characteristic = 'trial_survey_completeness_wrong_4'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 500, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Shoddy-Schechter' , lum_function_inf='Shoddy-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_wrong_5'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 500, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Full-Schechter' , lum_function_inf='Full-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'trial_survey_completeness_wrong_5_1'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 500, beta=-1.3, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Full-Schechter' , lum_function_inf='Full-Schechter', flux_threshold=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'trial_survey_completeness_right_2'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Shoddy-Schechter', lum_function_inf='Shoddy-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_2_corrected'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Shoddy-Schechter', lum_function_inf='Shoddy-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_2_L_star'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Shoddy-Schechter', lum_function_inf='Shoddy-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_3'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_3_L_star'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'trial_survey_completeness_right_3_RIGHT'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Random', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'trial_survey_completeness_right_4'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Shoddy-Schechter', lum_function_inf='Shoddy-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_4_L_star'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Shoddy-Schechter', lum_function_inf='Shoddy-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'trial_survey_completeness_right_5'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

# Corrected Inference
# Let's see how it does

investigated_characteristic = 'trial_survey_completeness_right_5_RIGHT'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
# Corrected Inference
# Let's see how it does

investigated_characteristic = 'trial_survey_completeness_right_5_RIGHT_1'
#investigated_values = [25,75,95]
investigated_values = np.array([0.05,0.1,0.2,0.3]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, beta=-1.3, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', luminosity_gen_type='Full-Schechter', lum_function_inf='Full-Schechter', flux_threshold=1, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    percentage.append(Investigation.survey_percentage)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'clustering'
investigated_values = [0, 10, 30, 50, 70, 90]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 200, coord_gen_type='Clustered', cluster_coeff=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution ='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'delta_D'
investigated_values = [4.035, 6.382, 13.804, 28.878]
rel = [5, 10, 20, 30]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, p_det=True, BVM_c=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%
    
investigated_characteristic = 'delta_D'
investigated_values = [2.964, 3.409, 4.953, 8.827, 74.244]
rel = [40, 35, 25, 15, 2]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, p_det=True, BVM_c=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)



#%%
    
'''
investigated_characteristic = 'event_num_log_rigorous'
investigated_values = [5, 7, 10, 15, 20, 40, 80, 160]
max_numbers = []
#b = []
#f = []

for i in tqdm(range(len(investigated_values))):
    Investigation = Sampler(universe_count = 100, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = investigated_values[i], specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)
'''

#%%

investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                            wanted_gal_n = investigated_values[i], specify_gal_number = True,
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'redshift_uncertainty_trash'
investigated_values = [0.005]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, d_ratio=0.6, beta=-1.3, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional',
                            total_luminosity = 500/3, specify_gal_number=True, wanted_gal_n=2000, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
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
max_numbers = ['0']



#%%

investigated_characteristic = 'trial_survey_completeness_0'
#investigated_values = [25,75,95]
investigated_values = np.array([1.0]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
investigated_values = [1.2732395447351628e-06]
max_numbers = ['0']*2

#%%

def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

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

p_i_s = []
c_i_s = []

for i in range(len(investigated_values)):
    #print(i)
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    means = []
    stds = []
    inc = df.index[1]-df.index[0]
    p_i_s.append(bias_dist(df))
    c_i_s.append(C_I_samp(df))
    for column in df.columns:
        if int(column)>999:
            continue
        if is_unique(df[column]):
            print('Gotcha')
            continue
        pdf_single = df[column]/(inc * df[column].sum())
        #print(df[column].sum())
        pdf_single.dropna(inplace=True)
        vals = np.array(pdf_single.index)
        mean = sum(inc * pdf_single*vals)
        # means or modes
        #mean = vals[np.argmax(pdf_single*vals)]
        if mean==0:
            continue
        means.append(mean)
        stds.append(np.sqrt(sum((inc*pdf_single*pdf_single.index**2))-mean**2))
    meanss.append(means)
    stdss.append(stds)
    pos.append(i+1)

ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

ax1.violinplot(meanss, bw_method=0.4, vert=False, showmeans=True)
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

#%%

N = 500
ci = np.linspace(1/N,1,N)

'''
significance = [1,2,3]
for i in range(len(significance)): 
    df_sig = pd.DataFrame(index=df.index)
    for j in range(100):
        #h = np.random.uniform(65,75)# + 5*significance[i])
        h = 70 + 5*significance[i]
        df_sig[str(j)] = sps.norm.pdf(df.index, loc=h, scale=5)

c_i_s_signif = C_I_samp(df_sig)
'''

for i in range(len(investigated_values)):
    fraction = []
    for j in ci:    
        fraction.append(sum(k<j for k in c_i_s[i])/len(c_i_s[i]))
    plt.stairs(fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])

'''
sigs_fraction = []
for j in ci:    
        sigs_fraction.append(sum(k<j for k in c_i_s_signif)/len(c_i_s_signif))
plt.stairs(sigs_fraction, np.insert(ci, 0, 0), lw=3, label=investigated_values[i])
'''

plt.legend()
plt.show()

#%%

for i in range(len(investigated_values)):
    plt.hist(p_i_s[i], bins=np.linspace(0,1,10), density=0, histtype='step', lw=3, label=investigated_values[i])
plt.legend()
plt.show()

#%%
    
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
B = 15
bins = np.linspace(0,1,B+1)
NN = len(p_i_s[0])
u_p = 0.995
d_p = 0.005

c = ['b', 'r']
vals = ['Not Included', 'Included']
for i in range(len(investigated_values)):
    ax.hist(p_i_s[i], bins=bins, density=0, linewidth=5, histtype='step',edgecolor=c[i], lw=4, label=vals[i])

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

upper_band = sps.binom.ppf(u_p, NN, 1/B)
lower_band = sps.binom.ppf(d_p, NN, 1/B)

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='gray', ls='dashed', lw=2)
ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='gray', alpha=0.5, label=r'${:.0f}\%$ confidence band'.format(100*(u_p-d_p)))
#ax.set_title('{} Average Events'.format(selection_accounted[j]))
ax.set_xlim(-0.003,1.003)
#ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.set_xlabel('Rank Statistic', fontsize=40, labelpad=15)
plt.legend(fontsize=30)
plt.show()

#%%

   
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
B = 10
bins = np.linspace(0,1,B+1)
NN = len(p_i_s[0])
u_p = 0.995
d_p = 0.005

c = ['b', 'r', 'magenta']
color = iter(cm.winter_r(np.linspace(0, 1, 3)))
for i in range(len(investigated_values)):
    c = next(color)
    ax.hist(p_i_s[i], bins=bins, density=0, linewidth=5, histtype='step',edgecolor=c, lw=4, label=r'$\sigma_z = {}$'.format(investigated_values[i]))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

upper_band = sps.binom.ppf(u_p, NN, 1/B)
lower_band = sps.binom.ppf(d_p, NN, 1/B)

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax.set_yticks([100,200,300,400,500])
ax.set_yticklabels(['100','200','300','400','500'])
ax.set_ylim(0,500)

ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='gray', ls='dashed', lw=2)
ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='gray', alpha=0.5, label=r'${:.0f}\%$ confidence band'.format(100*(u_p-d_p)))
#ax.set_title('{} Average Events'.format(selection_accounted[j]))
ax.set_xlim(-0.003,1.003)
#ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.set_xlabel('Rank Statistic', fontsize=40, labelpad=15)
plt.legend(fontsize=30)
plt.show()

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
B = 10
bins = np.linspace(0,1,B+1)
NN = len(p_i_s[0])
u_p = 0.995
d_p = 0.005

c = ['b', 'r', 'magenta']
color = iter(cm.winter_r(np.linspace(0, 1, 3)))
for i in range(len(investigated_values)):
    c = next(color)
    ax.hist(p_i_s[i], bins=bins, density=0, linewidth=5, histtype='step',edgecolor=c, lw=4, label=r'$f_{{\mathrm{{det}}}} = {:.1f} \%$'.format(per[i]))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

upper_band = sps.binom.ppf(u_p, NN, 1/B)
lower_band = sps.binom.ppf(d_p, NN, 1/B)

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax.set_yticks([50,100,150,200,250])
ax.set_yticklabels(['50','100','150','200','250'])
ax.set_ylim(0,250)
ax.hlines(y = [upper_band, lower_band], xmin=0, xmax=1, color='gray', ls='dashed', lw=2)
ax.fill_between(bins, [lower_band]*(B+1), [upper_band]*(B+1), color='gray', alpha=0.5, label=r'${:.0f}\%$ confidence band'.format(100*(u_p-d_p)))
#ax.set_title('{} Average Events'.format(selection_accounted[j]))
ax.set_xlim(-0.003,1.003)
#ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.set_xlabel('Rank Statistic', fontsize=40, labelpad=15)
plt.legend(fontsize=30)
plt.show()


#%%

def expected(data, sig):
    sig = np.array(sig)
    S = np.sum(1/sig**2)
    data = np.array(data)
    return np.sum(data/(sig**2))/S, np.sqrt(1/S)

for i in range(len(investigated_values)):
    bias, bias_err = expected(meanss[i], stdss[i])    
    s = 'H_0 = {:.2f}+/-{:.2f}'.format(bias, bias_err)
    print(s)


#%%













#%%

fig = plt.figure(figsize = (12,9.5), layout="constrained")
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax = [ax1,ax2]

ax1.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

for i in ax:
    i.set_ylim(0,0.49)
    i.set_xlim(65,80)

plt.subplots_adjust(wspace=0, hspace=0)

b = np.linspace(65,80,21)


#bias_0 = np.mean(meanss[0])-70
#bias_1 = np.mean(meanss[1])-70
#bias_err_0 = np.std(meanss[0])
#bias_err_1 = np.std(meanss[1])

bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])

bias_0 -= 70
bias_1 -= 70

ax1.hist(meanss[0], bins=b, histtype='step', edgecolor='magenta', lw=3.5, density=1)
ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='magenta', lw=3.5, density=1)
ax1.vlines(x=bias_0+70, ymin=0, ymax=0.49,ls='dashed', color='b', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49,ls='dashed', color='b', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1))
ax1.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax2.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)


#ax1.grid(axis='x', ls='dashed', alpha=0.5)
#ax2.grid(axis='x', ls='dashed', alpha=0.5)
ax1.legend(fontsize=25, framealpha=1, loc='upper left')
ax2.legend(fontsize=25, framealpha=1, loc='upper left')
ax1.set_xticklabels([])

incl = ['Not modelled', 'Modelled']
#ax2.set_ylabel('Included', fontsize=35, labelpad=15)
#ax1.set_ylabel('Not Included', fontsize=35, labelpad=15)
ax2.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=35, labelpad=15)
po = 74.5
ax1.annotate('{}'.format(incl[0]), xy=(po,0.3),xytext=(po,0.35), fontsize=35)
ax2.annotate('{}'.format(incl[1]), xy=(po,0.3),xytext=(po+0.7,0.35), fontsize=35)


fig.supylabel(r'$P\,(\hat H_0|\mathrm{samples})$', fontsize=40, x=-0.018)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)

#ax1.set_title('Selection effects', fontsize=40, pad=30)
#plt.tight_layout()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//selection_effects.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()

#%%


# %%

h = np.array(meanss[1])
sig = np.array(stdss[1])
S = np.sum(1/sig**2)
hat_h = np.sum(h/(sig**2))/S
print(np.mean(sig))
print(np.mean(sig)/np.sqrt(len(sig)))
print(np.sqrt(1/S))
print(np.std(h))
print(' ')
print(np.mean(h))
print(hat_h)








# %%

import numpy as np
import pandas as pd

title = 'Posterior Asymptotic Normality & Constraining Power'

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

 

#%%
    
#matplotlib.rcParams['font.family'] = 'Computer Modern Serif'

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

ax.set_ylim(outer_x_min,outer_x_max)
ax.set_xlim(plot_lim_min, plot_lim_max)

mus = []
sigs = []
maxs = []
argmaxs = []

space = N.tolist()
space.append(N[-1]*2)

axs = []
for i in range(len(df_N.columns)):
    ax1 = ax.inset_axes(
        [N[i], outer_x_min, (space[i+1]-space[i])/1.5, (outer_x_max-outer_x_min)], transform=ax.transData, zorder=1)
    axs.append(ax1)
for i in range(len(df_N.columns)):
    
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    df.dropna(inplace=True, axis=1)
    pdf_single = df[column]/df[column].sum() #* (df.index[1] - df.index[0])
    pdf_single.dropna(inplace=True)
    #vals = np.array(pdf_single.index)
    #mean = sum(pdf_single*vals)
    axs[i].spines[['right', 'top', 'bottom','left']].set_visible(False)   
    #n = random.choice(df.columns)
    #axs[i].plot(df[n], df.index, c='turquoise', lw=3.5)
    
    axs[i].plot(df_N[df_N.columns[i]], df_N.index, c='turquoise', lw=3.5)
    
    pdf = df_N[df_N.columns[i]]
    vals = np.array(pdf_single.index)
    max_p = np.max(pdf)
    argmax_p = vals[np.argmax(pdf)]
    mu = sum(pdf*vals)
    sig = np.sqrt(sum((pdf*vals**2))-mu**2)
    mus.append(mu)
    sigs.append(sig)
    maxs.append(max_p)
    argmaxs.append(argmax_p)
    adj = 1/(df_N.index[1] - df_N.index[0])
    #h = np.exp(-0.5)/(adj*np.sqrt(2*np.pi)*sig)
    #axs[i].vlines(x=h, ymax= mu+sig, ymin = mu-sig)
    #axs[i].hlines(y = [mu-sig,mu, mu+sig], xmax=h+h/2, xmin=h-h/2)
    ax.vlines(x=N[i], ymax= mu+sig, ymin = mu-sig)
    ax.hlines(y = [mu-sig,mu, mu+sig], xmax=N[i]+(space[i+1]-space[i])/8, xmin=N[i])
    axs[i].set_ylim(outer_x_min,outer_x_max)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    #axs[i].set_xlim(0,120)

x = N
y = sigs
x_cont = np.linspace(N[0]/rate,N[-1]*rate,1000)

def func(x,a):
    return a/np.sqrt(x)

popt, pcov = curve_fit(func, x, y)

ax.plot(x_cont, 70 + func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r', label=r'$\sigma = \alpha/\sqrt{{N}}$ fit, $\alpha={:.1f}\%\pm{:.1f}\%$'.format(100*popt[0]/70, 100*pcov[0,0]**0.5/70))
ax.plot(x_cont, 70 - func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r')
ax.plot([], [], c='turquoise', label='Average posteriors')
ax.plot([], [], c='white', label=r'$\mu\pm 1\sigma$')

ax.hlines(y=70, xmin=plot_lim_min, xmax=plot_lim_max, color='b')

ax.set_xscale('log')
N_labels = [str(i) for i in N]
ax.set_xticks(N)
ax.set_xticklabels(N_labels)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax.grid(axis='y', ls='dashed', alpha=0.7)
ax.set_xlabel('Average number of detected mergers', fontsize=35, labelpad=15)
ax.set_ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
#ax.set_title(title, x=0.46, fontsize=35, pad=30)
#ax.legend(fontsize=28, framealpha=1, loc=(0.357,0.705))
ax.legend(fontsize=28, framealpha=1, loc=(0.342,0.715))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//asymptotic_normality.svg'

plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()



#%%

   
#matplotlib.rcParams['font.family'] = 'Computer Modern Serif'

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

ax.set_ylim(outer_x_min,outer_x_max)
ax.set_xlim(plot_lim_min, plot_lim_max)

mus = []
sigs = []
maxs = []
argmaxs = []

space = N.tolist()
space.append(N[-1]*2)

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

axs = []
for i in range(len(df_N.columns)):
    ax1 = ax.inset_axes(
        [N[i], outer_x_min, (space[i+1]-space[i])/1.5, (outer_x_max-outer_x_min)], transform=ax.transData, zorder=1)
    axs.append(ax1)
for i in range(len(df_N.columns)):
  
    filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
    df = pd.read_csv(filename, index_col = 0)
    df.dropna(inplace=True, axis=1)
    pdf_single = df[column]/df[column].sum() #* (df.index[1] - df.index[0])
    pdf_single.dropna(inplace=True)
    #vals = np.array(pdf_single.index)
    #mean = sum(pdf_single*vals)
    axs[i].spines[['right', 'top', 'bottom','left']].set_visible(False)   
    n = random.choice(df.columns)
    axs[i].plot(df[n], df.index, c='turquoise', lw=3.5)
    
    #axs[i].plot(df_N[df_N.columns[i]], df_N.index, c='turquoise', lw=3.5)
    
    #axs[i].plot(df_N[df_N.columns[i]], df_N.index, c='turquoise', lw=3.5)
    
    pdf = df_N[df_N.columns[i]]
    vals = np.array(pdf_single.index)
    max_p = np.max(pdf)
    argmax_p = vals[np.argmax(pdf)]
    mu = sum(pdf*vals)
    sig = np.sqrt(sum((pdf*vals**2))-mu**2)
    
    mu,_ = expected(meanss[i], stdss[i])
    sig = np.mean(stdss[i])
    
    mus.append(mu)
    sigs.append(sig)
    maxs.append(max_p)
    argmaxs.append(argmax_p)
    adj = 1/(df_N.index[1] - df_N.index[0])
    #h = np.exp(-0.5)/(adj*np.sqrt(2*np.pi)*sig)
    #axs[i].vlines(x=h, ymax= mu+sig, ymin = mu-sig)
    #axs[i].hlines(y = [mu-sig,mu, mu+sig], xmax=h+h/2, xmin=h-h/2)
    #ax.vlines(x=N[i], ymax= mu+sig, ymin = mu-sig, lw=2, color='black')
    #ax.hlines(y = [mu-sig,mu, mu+sig], xmax=N[i]+(space[i+1]-space[i])/8, xmin=N[i], lw=2, color='black')
    axs[i].set_ylim(outer_x_min,outer_x_max)
    axs[i].set_yticklabels([])
    axs[i].set_xticklabels([])
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    #axs[i].set_xlim(0,120)

x = N
y = sigs
x_cont = np.linspace(N[0]/rate,N[-1]*rate,1000)

def func(x,a):
    return a/np.sqrt(x)

popt, pcov = curve_fit(func, x, y)

ax.plot(x_cont, 70 + func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r', label=r'$\sigma = \alpha/\sqrt{{N}}$ fit, $\alpha={:.1f}\%\pm{:.1f}\%$'.format(100*popt[0]/70, 100*pcov[0,0]**0.5/70))
ax.plot(x_cont, 70 - func(x_cont, *popt), ls='dashed', dashes=(5,5), lw=3, c='r')
ax.plot([], [], c='turquoise', label='Posteriors example')
#ax.plot([], [], c='white', label=r'$\mu\pm 1\sigma$')

ax.hlines(y=70, xmin=plot_lim_min, xmax=plot_lim_max, color='b', lw=3)

ax.set_xscale('log')
N_labels = [str(i) for i in N]
ax.set_xticks(N)
ax.set_xticklabels(N_labels)
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax.grid(axis='y', ls='dashed', alpha=0.7)
ax.set_xlabel('Average number of detected mergers', fontsize=35, labelpad=15)
ax.set_ylabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)
#ax.set_title(title, x=0.46, fontsize=35, pad=30)
#ax.legend(fontsize=28, framealpha=1, loc=(0.357,0.705))
ax.legend(fontsize=28, framealpha=1)#, loc=(0.342,0.715))

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//asymptotic_normality.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()

















# %%



from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator

#plt.style.use('default')

# Single event posterior plot
i = 0
filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
df = pd.read_csv(filename, index_col = 0)
dfp = df.prod(axis=1)
dfp /= dfp.sum()*(dfp.index[1]-dfp.index[0])



fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ymax=0.4
cmap = cm.get_cmap('winter')
color = iter(cm.winter(np.linspace(0, 1, len(df.columns))))
for i in range(len(df.columns)):
    c = next(color)
    ax.plot(df.index, df[str(i)], c=c, alpha=0.75)
    if i == int(len(df.columns)/3.01):
        mid_c = c
ax.plot([],[], c = mid_c, alpha=0.5, label='Single event posteriors')
ax.plot(dfp.index, dfp.values, c='magenta', lw=5, label='Full posterior')
ax.vlines(x=70, ymin=0, ymax=ymax, color='r', lw=3, ls='dashed', label='True value')


ax.set_xlim(50,100)
ax.set_ylim(0,ymax)
ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.tick_params(axis='both', which='major', direction='in', labelsize=35, size=10, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$P\,(\,H_0\, |\, d_{\mathrm{GW}}\,)$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)

#ax.set_xlim(59.9,80.1)
ax.set_ylim(-0.002,0.4)
ax.tick_params(axis='both', which='minor', direction='in', size=5, width=3, pad = 9)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//single_event.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()

#%%

from matplotlib.pyplot import cm
from matplotlib.ticker import MultipleLocator

#plt.style.use('default')

# Single event posterior plot
i = 0
filename = "PosteriorData/SampleUniverse_"+str(investigated_characteristic)+"_"+str(investigated_values[i])+"_"+max_numbers[i]+".csv"
df = pd.read_csv(filename, index_col = 0)
dfp = df.prod(axis=1)
dfp /= dfp.sum()*(dfp.index[1]-dfp.index[0])



fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

ymax=0.4
#cmap = cm.get_cmap('winter')
#color = iter(cm.winter(np.linspace(0, 1, len(df.columns))))
#for i in range(len(df.columns)):
#    c = next(color)
#    ax.plot(df.index, df[str(i)], c=c, alpha=0.75)
#    if i == int(len(df.columns)/3.01):
#        mid_c = c
#ax.plot([],[], c = mid_c, alpha=0.5, label='Single event posteriors')
ax.plot(dfp.index, dfp.values, c='dodgerblue', lw=5)#, label=r'$H_0$ Posterior')
#ax.vlines(x=70, ymin=0, ymax=ymax, color='r', lw=3, ls='dashed', label='True value')


ax.set_xlim(50,100)
ax.set_ylim(0,ymax)
ax.grid(axis='both', ls='dashed', c='lightblue', alpha=0.9)
ax.tick_params(axis='both', which='major', direction='in', labelsize=35, size=10, width=3, pad = 9)
ax.legend(fontsize = 28, framealpha=1)
ax.set_ylabel(r'$P\,(\,H_0\, |\, d_{\mathrm{GW}}\,)$', fontsize=35, labelpad=15)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=35, labelpad=15)

ax.set_xlim(59.9,80.1)
ax.set_ylim(-0.002,0.4)
ax.tick_params(axis='both', which='minor', direction='in', size=5, width=3, pad = 9)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//single_event.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()








# %%

# Redshift Uncertainty


fig = plt.figure(figsize = (12,10))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ax = [ax1,ax2, ax3]

ax1.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)
ax3.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax1.set_ylim(0,0.35)
ax1.set_xlim(50,90)
ax2.set_ylim(0,0.2)
ax2.set_xlim(50,90)
ax3.set_ylim(0,0.1)
ax3.set_xlim(50,90)

plt.subplots_adjust(wspace=0, hspace=0)

b = np.linspace(50,90,35)
b=16

bias_0 = np.mean(meanss[0])-70
bias_1 = np.mean(meanss[1])-70
bias_2 = np.mean(meanss[2])-70
bias_err_0 = np.std(meanss[0])
bias_err_1 = np.std(meanss[1])
bias_err_2 = np.std(meanss[2])


bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])
bias_2, bias_err_2 = expected(meanss[2], stdss[2])

bias_0 -= 70
bias_1 -= 70
bias_2 -= 70


ax1.hist(meanss[0], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)
ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)
ax3.hist(meanss[2], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)

ax1.vlines(x=bias_0+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1))
ax3.vlines(x=bias_2+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_2, bias_err_2))

ax1.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax2.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax3.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)


#ax1.grid(axis='x', ls='dashed', alpha=0.5)
#ax2.grid(axis='x', ls='dashed', alpha=0.5)
ax1.legend(fontsize=20)
ax2.legend(fontsize=20)
ax3.legend(fontsize=20)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax1.set_yticks([0.0,0.1,0.2, 0.3])
ax2.set_yticks([0.0, 0.07,0.14])
ax3.set_yticks([0.0,0.04,0.08])


#ax1.annotate(r'$\sigma_z = {}$'.format(investigated_values[0]), xy=(55,0.3),xytext=(52.5,0.3), fontsize=35)
#ax2.annotate(r'$\sigma_z = {}$'.format(investigated_values[1]), xy=(55,0.3*0.3/0.49),xytext=(52.5,0.3*0.3/0.49), fontsize=35)
#ax3.annotate(r'$\sigma_z = {}$'.format(investigated_values[2]), xy=(55,0.08 *0.3/0.49),xytext=(52.5,0.08*0.3/0.49), fontsize=35)


ax1.annotate(r'$\sigma_z = {}$'.format(investigated_values[0]), xy=(55,0.3),xytext=(52.5,0.2), fontsize=35)
ax2.annotate(r'$\sigma_z = {}$'.format(investigated_values[1]), xy=(55,0.3*0.3/0.49),xytext=(52.5,0.1), fontsize=35)
ax3.annotate(r'$\sigma_z = {}$'.format(investigated_values[2]), xy=(55,0.08 *0.3/0.49),xytext=(52.5,0.05), fontsize=35)


ax3.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=35, labelpad=15)

#ax1.set_title('Redshift uncertainty', fontsize=40, pad=30)

ax2.set_ylabel(r'$P\,(\hat H_0|\mathrm{samples})$', fontsize=40, labelpad=20)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)
    ax3.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//redshift_uncertainty.svg'

#plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()






# %%

# %%

# Survey completeness


fig = plt.figure(figsize = (12,10))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ax = [ax1,ax2, ax3]

ax1.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)
ax3.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 12)

ax1.set_ylim(0,0.49)
ax1.set_xlim(62.6,77.4)
ax2.set_ylim(0,0.4)
ax2.set_xlim(62.6,77.4)
ax3.set_ylim(0,0.4)
ax3.set_xlim(62.6,77.4)

plt.subplots_adjust(wspace=0, hspace=0)

b = 15
bias_0 = np.mean(meanss[0])-70
bias_1 = np.mean(meanss[1])-70
bias_2 = np.mean(meanss[2])-70
bias_err_0 = np.std(meanss[0])
bias_err_1 = np.std(meanss[1])
bias_err_2 = np.std(meanss[2])


bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])
bias_2, bias_err_2 = expected(meanss[2], stdss[2])

bias_0 -= 70
bias_1 -= 70
bias_2 -= 70

ax1.hist(meanss[0], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)
ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)
ax3.hist(meanss[2], bins=b, histtype='step', edgecolor='magenta', lw=3, density=1)

ax1.vlines(x=bias_0+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1))
ax3.vlines(x=bias_2+70, ymin=0, ymax=0.49, color='b',ls='dashed', lw=3, label='$\langle\hat H_0-H_0\\rangle={:.2f}\pm{:.2f}$'.format(bias_2, bias_err_2))

ax1.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax2.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax3.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)


#ax1.grid(axis='x', ls='dashed', alpha=0.5)
#ax2.grid(axis='x', ls='dashed', alpha=0.5)
ax1.legend(fontsize=20)
ax2.legend(fontsize=20)
ax3.legend(fontsize=20)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax2.set_yticks([0.0,0.15,0.30])
ax3.set_yticks([0.0,0.15,0.30])


#per = []
#for i in range(len(percentage)):
#    per.append(100*(np.mean(percentage[i])))

'''
ax1.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[0]), xy=(53,0.3),xytext=(52.5,0.3), fontsize=35)
ax2.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[1]), xy=(53,0.4*0.3/0.49),xytext=(52.5,0.4*0.3/0.49), fontsize=35)
ax3.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[2]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.3/0.49), fontsize=35)

ax1.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[0]), xy=(53,0.3),xytext=(52.5,0.17), fontsize=35)
ax2.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[1]), xy=(53,0.4*0.3/0.49),xytext=(52.5,0.4*0.17/0.49), fontsize=35)
ax3.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[2]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.17/0.49), fontsize=35)
'''

ax1.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[0]), xy=(63,0.3),xytext=(63.5,0.3), fontsize=35)
ax2.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[1]), xy=(63,0.4*0.3/0.49),xytext=(63.5,0.4*0.3/0.49), fontsize=35)
ax3.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[2]), xy=(63,0.08 *0.3/0.49),xytext=(63.5,0.4*0.3/0.49), fontsize=35)

ax1.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[0]), xy=(63,0.3),xytext=(63.5,0.17), fontsize=35)
ax2.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[1]), xy=(63,0.4*0.3/0.49),xytext=(63.5,0.4*0.17/0.49), fontsize=35)
ax3.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[2]), xy=(63,0.08 *0.3/0.49),xytext=(63.5,0.4*0.17/0.49), fontsize=35)


ax3.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=35, labelpad=15)

ax2.set_ylabel(r'$P\,(\hat H_0|\mathrm{samples})$', fontsize=40, labelpad=20)

#ax1.set_title('Survey completeness', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)
    ax3.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//survey_completeness_many.svg'

# plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()


# %%

# Survey completeness


fig = plt.figure(figsize = (12,10))
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=4,
                         wspace=0.2,
                         hspace=0.)

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
ax4 = fig.add_subplot(spec[3])
ax = [ax1,ax2, ax3, ax4]

for i in ax:
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
    i.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax1.set_ylim(0,0.49)
ax1.set_xlim(50,90)
ax2.set_ylim(0,0.4)
ax2.set_xlim(50,90)
ax3.set_ylim(0,0.2)
ax3.set_xlim(50,90)
ax4.set_ylim(0,0.2)
ax4.set_xlim(50,90)

plt.subplots_adjust(wspace=0, hspace=0)

b = 12
bias_0 = np.mean(meanss[0])-70
bias_1 = np.mean(meanss[1])-70
bias_2 = np.mean(meanss[2])-70
bias_3 = np.mean(meanss[3])-70
bias_err_0 = np.std(meanss[0])
bias_err_1 = np.std(meanss[1])
bias_err_2 = np.std(meanss[2])
bias_err_3 = np.std(meanss[3])


bias_0, bias_err_0 = expected(meanss[0], stdss[0])
bias_1, bias_err_1 = expected(meanss[1], stdss[1])
bias_2, bias_err_2 = expected(meanss[2], stdss[2])
bias_3, bias_err_3 = expected(meanss[3], stdss[3])


bias_0 -= 70
bias_1 -= 70
bias_2 -= 70
bias_3 -= 70

ax1.hist(meanss[0], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1)
ax2.hist(meanss[1], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1)
ax3.hist(meanss[2], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1)
ax4.hist(meanss[3], bins=b, histtype='step', edgecolor='turquoise', lw=3, density=1)

ax1.vlines(x=bias_0+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_0, bias_err_0))
ax2.vlines(x=bias_1+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_1, bias_err_1))
ax3.vlines(x=bias_2+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_2, bias_err_2))
ax4.vlines(x=bias_3+70, ymin=0, ymax=0.49,ls='dashed', lw=3, label='$\hat H_0-H_0={:.2f}\pm{:.2f}$'.format(bias_3, bias_err_3))

ax1.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax2.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax3.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)
ax4.vlines(x=70, ymin=0, ymax=0.49, color='r', ls='dashed', lw=3)


#ax1.grid(axis='x', ls='dashed', alpha=0.5)
#ax2.grid(axis='x', ls='dashed', alpha=0.5)
ax1.legend(fontsize=20)
ax2.legend(fontsize=20)
ax3.legend(fontsize=20)
ax4.legend(fontsize=20)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax2.set_yticks([0.0,0.15,0.30])
ax3.set_yticks([0.0,0.08,0.16])
ax4.set_yticks([0.0,0.08,0.16])

'''
per = []
for i in range(len(percentage)):
    per.append(100*(np.mean(percentage[i])))
'''

ax1.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[0]), xy=(53,0.3),xytext=(52.5,0.3), fontsize=35)
ax2.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[1]), xy=(53,0.4*0.3/0.49),xytext=(52.5,0.4*0.3/0.49), fontsize=35)
ax3.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[2]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.3/0.49), fontsize=35)
ax4.annotate(r'$F_{{\mathrm{{lim}}}}/F_* = {}$'.format(l_lim[3]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.3/0.49), fontsize=35)

ax1.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[0]), xy=(53,0.3),xytext=(52.5,0.17), fontsize=35)
ax2.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[1]), xy=(53,0.4*0.3/0.49),xytext=(52.5,0.4*0.17/0.49), fontsize=35)
ax3.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[2]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.17/0.49), fontsize=35)
ax4.annotate(r'$f_{{\mathrm{{det}}}} = {:.0f}\%$'.format(per[3]), xy=(53,0.08 *0.3/0.49),xytext=(52.5,0.2*0.17/0.49), fontsize=35)

ax4.set_xlabel(r'$\hat H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=35, labelpad=15)

ax1.set_title('Survey completenes Right', fontsize=40, pad=30)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(3)
    ax2.spines[axis].set_linewidth(3)
    ax3.spines[axis].set_linewidth(3)
    ax4.spines[axis].set_linewidth(3)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//survey_completeness_many.svg'

# plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()

# %%
