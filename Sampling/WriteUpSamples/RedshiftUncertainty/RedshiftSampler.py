#%%

import sys
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler


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

investigated_characteristic = 'redshift_uncertainty_incorrect'
investigated_values = [0.005]
investigated_values = [0.01]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 1, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=500, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty_incorrect_3'
investigated_values = [0.005]
#investigated_values = [0.01]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 500, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=500, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty_incorrect_6'
investigated_values = [0.005]
#investigated_values = [0.01]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 50, beta=-1.3, lower_lim=0.1, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=5000, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty_incorrect_5'
investigated_values = [0.005]
#investigated_values = [0.01]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, survey_type='perfect', redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=3000, wanted_det_events = 16, d_ratio=0.6, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)


#%%

investigated_characteristic = 'redshift_uncertainty_standard_incorrect'
investigated_values = [0.005]
investigated_characteristic = 'redshift_uncertainty_standard_incorrect_2'
investigated_values = [0.01]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 50, survey_type='perfect', d_ratio=0.6, beta=-1.3, lower_lim=0.1, redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Random', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=4000, wanted_det_events = 20, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Random', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
    Investigation.Sample()
    #b.append(Investigation.burr_i)
    #f.append(Investigation.full)
    max_numbers.append(Investigation.max_num)

#%%

investigated_characteristic = 'redshift_uncertainty_standard_incorrect_3'
investigated_values = [0.005]
max_numbers = []
#b = []
#f = []

for i in range(len(investigated_values)):
    Investigation = Sampler(universe_count = 100, survey_type='perfect', d_ratio=0.7, beta=-1.3, lower_lim=0.1, redshift_noise_sigma=investigated_values[i], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=3000, wanted_det_events = 10, specify_event_number = True, 
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i],
                            save_normally=0)
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
    

investigated_characteristic = 'redshift_uncertainty_corrected_approx_additional_run'
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