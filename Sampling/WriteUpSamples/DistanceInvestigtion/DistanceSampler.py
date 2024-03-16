#%%

import sys
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler

#%%

# Distance Unertainty

investigated_characteristic = 'delta_D'
investigated_values = [2.964, 3.409, 4.953, 8.827] + [4.035, 6.382, 13.804, 28.878]
investigated_values.sort()
rel = [40, 35, 25, 15] + [5, 10, 20, 30]
rel.sort()
Ns = [5, 10, 20, 50, 100, 200]
max_numbers = []
#b = []
#f = []

for j in Ns:
    investigated_characteristic = 'delta_D' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, BVM_c=investigated_values[i], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True, 
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)


#%%
    
investigated_characteristic = 'delta_D'
investigated_values = [2.637, 2.3902]
rel = [45, 50]
rel.sort()
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

#Galaxy Number

investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000, 16000]
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

investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000, 16000]
max_numbers = []
#b = []
#f = []
    
Ns = [5, 10, 20, 50, 100, 200]
true_Ns = []

for j in Ns:
    investigated_characteristic = 'gal_num_set' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = investigated_values[i], specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

#%%
        
# Save the true N values


# %%

investigated_characteristic = 'gal_num_set'
investigated_values = [250, 500, 1000, 2000, 4000, 8000]
max_numbers = []
#b = []
#f = []
    
Ns = [200]
true_Ns = []

for j in Ns:
    investigated_characteristic = 'gal_num_set' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = investigated_values[i], specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

# %%
