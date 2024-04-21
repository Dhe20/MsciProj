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
    investigated_characteristic = 'lum_weighting' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        if investigated_values[i] == investigated_values_inference[i]:
            r_or_w = 'right'
        else:
            r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution=investigated_values[i], total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = 2000, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf=investigated_values_inference[i], 
                                save_normally=0, investigated_characteristic = investigated_characteristic + r_or_w, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

#%%
        

investigated_characteristic = 'lum_weighting_pres'
investigated_values = ['Proportional', 'Proportional', 'Random', 'Random']
investigated_values_inference = ['Proportional', 'Random', 'Random', 'Proportional']
max_numbers = []
#b = []
#f = []
    
Ns = [2,4,8,16,32,64,128]
true_Ns = []

for j in Ns:
    investigated_characteristic = 'lum_weighting' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        if investigated_values[i] == investigated_values_inference[i]:
            r_or_w = 'right'
        else:
            r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution=investigated_values[i], total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = 5000, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf=investigated_values_inference[i], 
                                save_normally=0, investigated_characteristic = investigated_characteristic + r_or_w, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'lum_weighting_001'
investigated_values = ['Proportional', 'Proportional', 'Random']
investigated_values_inference = ['Proportional', 'Proportional', 'Random']
betas = [-1.05,-1.95,-1.95]
max_numbers = []

investigated_characteristic = 'lum_weighting_01'
investigated_values = ['Proportional', 'Proportional', 'Random']
investigated_values_inference = ['Proportional', 'Proportional', 'Random']
betas = [-1.05,-1.95,-1.95]
max_numbers = []

#b = []
#f = []
    
Ns = [2,4,6,8,10,12,16,24,32,64,128]
#Ns = [24,32,64,128]


true_Ns = []

for j in Ns:
    #investigated_characteristic = 'lum_weighting' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        investigated_characteristic = 'lum_weighting_001'+'_'+str(-1*betas[i])+'beta' + '_' + str(j) + '_' + 'average_events'
        print(investigated_characteristic)
        #if investigated_values[i] == investigated_values_inference[i]:
        #    r_or_w = 'right'
        #else:
        #    r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 100, beta=betas[i], lower_lim=0.1, characteristic_luminosity=1, p_det=True, gamma = False, event_distribution=investigated_values[i], total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = 7500, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf=investigated_values_inference[i], 
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'lum_weighting_standard_nice'
investigated_values = ['Proportional', 'Proportional', 'Random']
investigated_values_inference = ['Proportional', 'Proportional', 'Random']
betas = [-1.05,-1.95,-1.95]
max_numbers = []

#b = []
#f = []

Ns = [2,4,6,8,10,12,16,24,32,64,128]
#Ns = [24,32,64,128]


true_Ns = []

for j in Ns:
    #investigated_characteristic = 'lum_weighting' + '_' + str(j) + '_' + 'average_events'
    for i in range(len(investigated_values)):
        investigated_characteristic = 'lum_weighting_standard_nice'+'_'+str(-1*betas[i])+'beta' + '_' + str(j) + '_' + 'average_events'
        print(investigated_characteristic)
        #if investigated_values[i] == investigated_values_inference[i]:
        #    r_or_w = 'right'
        #else:
        #    r_or_w = 'wrong'
        Investigation = Sampler(universe_count = 200, beta=betas[i], lower_lim=0.1, characteristic_luminosity=1, p_det=True, gamma = False, event_distribution=investigated_values[i], total_luminosity=1000/3, wanted_det_events = j, specify_event_number = True,
                                wanted_gal_n = 7500, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf=investigated_values_inference[i], 
                                save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)


#%%
        
# Save the true N values
        
import pickle
name = investigated_characteristic + '_' + 'Ns'
with open(name, "wb") as fp:   #Pickling
    pickle.dump(true_Ns, fp)

with open(name, "rb") as fp:   # Unpickling
    b = pickle.load(fp)



#%%
        
# Selection Effects
        
investigated_characteristic = 'D_max_ratio'
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
                                wanted_gal_n = 2000, specify_gal_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                                save_normally=0, investigated_characteristic = investigated_characteristic + r_or_w, investigated_value = investigated_values[i])
        Investigation.Sample()
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)

# %%


# Selection Effects
        
investigated_characteristic = 'sel_eff_sigma_D'
investigated_values = [3.409, 4.035, 4.953, 6.382, 8.827, 13.804, 28.878, 49.003]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [3, 5, 10, 15, 20, 25, 30, 35]
max_numbers = []

investigated_characteristic = 'sel_eff_sigma_D'
investigated_values = [7.423, 10.81, 18.817]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [7.5, 12.5, 17.5]
max_numbers = []

#%%

investigated_characteristic = 'sel_eff_sigma_D_standard_unaccounted'
investigated_values = [6.382, 8.827, 13.804, 28.878, 49.033] + [7.423, 10.81, 18.817]
investigated_values = [9.533]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [7.5, 12.5, 17.5] + [3, 5, 10, 15, 20]
rel = [14]
rel.sort()
max_numbers = []
print(rel)
print(investigated_values)


#%%

investigated_characteristic = 'sel_eff_sigma_D_standard_unaccounted_120'
investigated_values = [6.382, 8.827, 13.804, 28.878, 49.033] + [7.423, 10.81, 18.817] + [5.582] + [4.953]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [7.5, 12.5, 17.5] + [3, 5, 10, 15, 20] + [22.5] + [25]
rel.sort()
max_numbers = []
print(rel)
print(investigated_values)


#b = []
#f = []

#%%

investigated_characteristic = 'sel_eff_sigma_D_standard_unaccounted_0.3'
investigated_values = [6.382, 8.827, 13.804, 28.878, 49.033] + [7.423, 10.81, 18.817]
#investigated_values = [9.533]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [7.5, 12.5, 17.5] + [3, 5, 10, 15, 20]
#rel = [14]
rel.sort()
max_numbers = []
print(rel)
print(investigated_values)

#%%

true_Ns = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, d_ratio=0.3, BVM_c = investigated_values[i], beta=-1.3, lower_lim=0.1, p_det=False, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                            wanted_gal_n = 5000, specify_gal_number = True,
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                            save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    true_Ns.append(Investigation.det_event_count_for_analysis)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'sel_eff_sigma_D_biased_standard_140'
investigated_values = [3.409, 4.035, 4.953, 6.382, 8.827, 13.804, 28.878, 49.003]
investigated_values.sort(reverse=True)
#selection_accounted = [True, False]
rel = [3, 5, 10, 15, 20, 25, 30, 35]
max_numbers = []
#b = []
#f = []

true_Ns = []

for i in range(len(investigated_values)):
    print(investigated_characteristic)
    Investigation = Sampler(universe_count = 100, H_0_Max=140, BVM_c = investigated_values[i], beta=-1.3, lower_lim=0.1, p_det=False, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                            wanted_gal_n = 5000, specify_gal_number = True,
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', 
                            save_normally=0, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
    Investigation.Sample()
    true_Ns.append(Investigation.det_event_count_for_analysis)
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

# %%
