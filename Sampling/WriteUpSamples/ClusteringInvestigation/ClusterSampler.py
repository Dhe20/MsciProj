#%%
from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Sampling.ClassSamples import Sampler

universe_count = 10
dimension = 3
rate = 10**6
wanted_det_events = 50

characteristic_Luminosity = 1
total_luminosity = 1000
H_0_Min = 50
H_0_Max = 100
resolution_H_0 = 100
size = 625

investigated_characteristic = "clustering"
investigated_values = [5,10,20,30,40,50]

for i in range(0,len(investigated_values)):
    Investigation = Sampler(universe_count = 100, BVM_c=6, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3,
                            wanted_det_events = 20, specify_event_number = True,
                            coord_gen_type="Clustered", cluster_coeff=investigated_values[i],
                            noise_distribution='BVMF_eff', event_distribution_inf='Proportional', lum_function_inf='Full-Schechter',
                            investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i], save_normally=False)
    Investigation.Sample()


# %%
