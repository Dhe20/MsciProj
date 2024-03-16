#%%

import sys
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler

#%%

investigated_characteristic = 'lum_weighting'
investigated_values = ['Proportional', 'Proportional', 'Random', 'Random']
investigated_values_inference = ['Proportional', 'Random', 'Random', 'Proportional']
max_numbers = []
#b = []
#f = []
    
Ns = [5, 10, 20, 50, 100, 200]
true_Ns = []

for j in Ns:
    investigated_characteristic = 'gal_num_set' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        if investigated_values[i] == investigated_values_inference[i]:
            r_or_w = 'right'
        else:
            r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution=investigated_values[i], total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = 4000, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf=investigated_values_inference[i], 
                                save_normally=0, investigated_characteristic = investigated_characteristic + r_or_w, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

#%%
        
# Save the true N values
        




#%%
        
# Selection Effects
        
investigated_characteristic = 'lum_weighting'
investigated_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
selection_accounted = [True, False]
max_numbers = []
#b = []
#f = []

true_Ns = []

for i in range(len(investigated_values)):
    for k in range(len(selection_accounted)):
        if selection_accounted[k]:
            r_or_w = 'right'
        else:
            r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 100, d_ratio = investigated_values[i], beta=-1.3, p_det=selection_accounted[k], gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                                wanted_gal_n = 8000, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                                save_normally=0, investigated_characteristic = investigated_characteristic + r_or_w, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

# %%
