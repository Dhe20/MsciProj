#%%
from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Sampling.ClassSamples import Sampler

universe_count = 500
dimension = 3
rate = 10**6
wanted_det_events = 50

characteristic_luminosity = 1
total_luminosity = 5000/3
H_0_Min = 50
H_0_Max = 100
resolution_H_0 = 200
size = 625

investigated_characteristic = "clustering"
investigated_values = [0.01,1,2,3,4,5,6,7,8,9,10]

N_gals = [625/12]

for N_gal in N_gals:
    for i in range(0,len(investigated_values)):
        Investigation = Sampler(universe_count = 500, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=N_gal,
                                characteristic_luminosity=characteristic_luminosity, resolution_H_0 = resolution_H_0, H_0_Min = H_0_Min, H_0_Max = H_0_Max,
                                wanted_det_events = 50, specify_event_number = True,
                                coord_gen_type="Clustered", cluster_coeff=investigated_values[i],
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', lum_function_inf='Full-Schechter',
                                investigated_characteristic = investigated_characteristic +"_" +str(round(N_gal*3,1)), investigated_value = investigated_values[i], save_normally=False, start_seed = 0)
        Investigation.Sample()


# %%
