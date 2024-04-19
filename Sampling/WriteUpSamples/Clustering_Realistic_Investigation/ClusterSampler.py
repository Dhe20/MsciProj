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

characteristic_luminosity = .1
total_luminosity = 500/3
H_0_Min = 50
H_0_Max = 100
resolution_H_0 = 200
size = 625

investigated_characteristics = ["10LocVol_realistic_clustering", "20LocVol_realistic_clustering"]
kappas = [200, 20]
cs = [15, 6.382]


# for i, coeff in enumerate([0.01,0.05, 0.1,0.5,1, 5]):
# for i, coeff in enumerate([0.0005, 0.001, 0.005, 0.1,0.5,1, 5]):
for i, investigated_characteristic in enumerate(investigated_characteristics):
    if i == 0:
        continue
    for j, coeff in enumerate([0.0005, 0.001, 0.005 ,0.01,0.05, 0.1,0.5,1, 5]):
        Investigation = Sampler(dimension=dimension,
                                universe_count = universe_count, p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=total_luminosity,
                                characteristic_luminosity=characteristic_luminosity, resolution_H_0 = resolution_H_0, H_0_Min = H_0_Min, H_0_Max = H_0_Max,
                                wanted_det_events = 50, specify_event_number = True, BVM_c = cs[i], BVM_k=2, BVM_kappa=kappas[i],
                                coord_gen_type="Clustered", cluster_coeff=coeff,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', lum_function_inf='Full-Schechter',
                                investigated_characteristic = investigated_characteristic, investigated_value = coeff, save_normally=False, start_seed = 0,
                                log_event_count = True)
        Investigation.Sample()

