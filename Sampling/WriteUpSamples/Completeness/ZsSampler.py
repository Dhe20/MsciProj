#%%

import sys
import numpy as np
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler

#%%

investigated_characteristic = 'redshift_uncertainty'
investigated_values = [0.0005 ,0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

investigated_characteristic = 'redshift_uncertainty_more_5000_standard_good'
investigated_values = [0.0005 ,0.001, 0.002, 0.004, 0.006, 0.01]

#investigated_characteristic = 'redshift_uncertainty_more_2500_standard'
#investigated_values = [0.0005 ,0.001, 0.002, 0.004, 0.006, 0.01]

investigated_characteristic = 'redshift_uncertainty_more_2500_standard_good'
investigated_values = [0.0005 ,0.001, 0.002, 0.004, 0.006, 0.01]


max_numbers = []
true_Ns = []

#b = []
#f = []

for i in range(len(investigated_values)):
    
    Investigation = Sampler(universe_count = 100, beta=-1.3, lower_lim=0.1, redshift_noise_sigma=investigated_values[i], p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                                wanted_gal_n = 2500, specify_gal_number = True, noise_distribution='BVMF_eff', event_distribution_inf='Proportional',
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])

    Investigation.Sample()
    true_Ns.append(Investigation.det_event_count_for_analysis)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


# %%

investigated_characteristic = 'redshift_uncertainty_more_5000_standard_good_random'
investigated_values = [0.0005 ,0.001, 0.002, 0.004, 0.006, 0.01]

max_numbers = []
true_Ns = []

#b = []
#f = []

for i in range(len(investigated_values)):
    
    Investigation = Sampler(universe_count = 100, beta=-1.3, lower_lim=0.1, redshift_noise_sigma=investigated_values[i], p_det=True, gamma = False, event_distribution='Random', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                                wanted_gal_n = 5000, specify_gal_number = True, noise_distribution='BVMF_eff', event_distribution_inf='Random',
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])

    Investigation.Sample()
    true_Ns.append(Investigation.det_event_count_for_analysis)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty_more_2500_standard_good_random'
investigated_values = [0.0005 ,0.001, 0.002, 0.004, 0.006, 0.01]

max_numbers = []
true_Ns = []

#b = []
#f = []

for i in range(len(investigated_values)):
    
    Investigation = Sampler(universe_count = 100, beta=-1.3, lower_lim=0.1, redshift_noise_sigma=investigated_values[i], p_det=True, gamma = False, event_distribution='Random', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                                wanted_gal_n = 2500, specify_gal_number = True, noise_distribution='BVMF_eff', event_distribution_inf='Random',
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])

    Investigation.Sample()
    true_Ns.append(Investigation.det_event_count_for_analysis)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


# %%
