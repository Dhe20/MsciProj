#%%
import sys
import numpy as np
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler


#%%
        
# Completeness
        
investigated_characteristic = 'survey_completeness'
investigated_values = np.array([0.005,0.01,0.05,0.1,0.2,0.3,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
dist = ['Proportional', 'Random']
percentage_s = []
max_numbers = []
#b = []
#f = []

true_Ns = []

for k in range(len(dist)):
    percentage = []
    for i in range(len(investigated_values)):
        Investigation = Sampler(universe_count = 100, beta=-1.3, p_det=True, gamma = False, event_distribution=dist[k], total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True,
                                wanted_gal_n = 2000, specify_gal_number = True, min_flux=investigated_values[i], completeness_type='cut_lim',
                                noise_distribution='BVMF_eff', event_distribution_inf=dist[k], flux_threshold=0,
                                save_normally=0, investigated_characteristic = investigated_characteristic+ '_' + dist[k], investigated_value = investigated_values[i])
        Investigation.Sample()
        percentage.append(np.mean(Investigation.survey_percentage))
        true_Ns.append(Investigation.det_event_count_for_analysis)
        #b.append(Investigation.burr_i)
        #f.append(Investigation.full)
        max_numbers.append(Investigation.max_num)
    print(percentage)
    percentage_s.append(percentage)


#%%
        
print(percentage_s)

import pickle
l = [1,2,3,4]
with open("test", "wb") as fp:   #Pickling
    pickle.dump(l, fp)

with open("test", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

print(b)
# %%


import pickle

with open("percentages", "wb") as fp:   #Pickling
    pickle.dump(percentage_s, fp)

with open("percentages", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

# %%
