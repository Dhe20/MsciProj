from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Sampling.ClassSamples import Sampler

universe_count = 1000
dimension = 3
rate = 10**6
wanted_det_events = 50

characteristic_luminosity = .1
total_luminosity = 500/3
H_0_Min = 50
H_0_Max = 100
resolution_H_0 = 200
size = 625
centroid_n = 25

investigated_characteristic = "CentroidSigma"
# investigated_values = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]
investigated_values = [0.01]
# investigated_values = [0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.40]

for centroid_n in [10,15,20,25]:
    for i in range(0,len(investigated_values)):
        Investigation = Sampler(universe_count = universe_count, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=total_luminosity,
                                characteristic_luminosity=characteristic_luminosity, resolution_H_0 = resolution_H_0, H_0_Min = H_0_Min, H_0_Max = H_0_Max,
                                wanted_det_events = 50, specify_event_number = True,
                                coord_gen_type="Centroids", cluster_coeff=0, centroid_n=centroid_n, centroid_sigma=investigated_values[i],
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', lum_function_inf='Full-Schechter',
                                investigated_characteristic = investigated_characteristic +"_" +str(centroid_n), investigated_value = investigated_values[i], save_normally=False, start_seed = 0,
                                log_event_count = True)
        Investigation.Sample()
