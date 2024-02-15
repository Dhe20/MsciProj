#%%
import sys
import pandas as pd
import numpy as np
from Components.Inference import Inference
from Components.EventGenerator import EventGenerator
import random
import os
import re
from tqdm import tqdm

resolution_H_0 = 200
H_0_Min = 50
H_0_Max = 100
universe_count = 200
total_luminosity = 1000
rate = 1000

H_0_samples = pd.DataFrame()


# event_num = np.linspace(5,50,10).astype(int)
# times =event_num/(total_luminosity*rate*(1.2*32*np.pi/3000))

L0_R_T = 1487.77872778
times = L0_R_T/(total_luminosity*rate)
time = times

#Change these when you change the size of your universe
# min_flux_dict = {'0.1': 3.755907975748638e-13, '0.5': 4.3065235389892225e-12, '1': 1.5298079988990114e-11, '5': 3.281732822052792e-10, '10': 1.20952966205507e-09, '50': 3.5884837357211884e-08, '99': 1.408736495936212e-06}


characteristic_Luminosity = 1
size = 625
dimension = 3

characteristic_flux = characteristic_Luminosity/(4*np.pi*(size*0.4)**2)

keys = [0.01, 0.05, 0.1, 0.5, 1, 5]

min_flux_dict = {}

for key in keys:
    min_flux_dict[str(key)] = key*characteristic_flux

min_fluxes = list(min_flux_dict.values())
min_flux_pct = list(min_flux_dict.keys())

for Universe in tqdm(range(universe_count)):


    H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

    Gen = EventGenerator(dimension=dimension, size=size, event_rate=rate, sample_time=time,
                         luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                         cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity,
                         total_luminosity=total_luminosity,
                         event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                         resolution=100, plot_contours=False, alpha=0.3, seed=Universe)

    Data = Gen.GetSurveyAndEventData()
    I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type="perfect")
    H_0_sample = I.H_0_Prob()
    H_0_samples[Universe] = H_0_sample

    for i, min_flux in enumerate(min_fluxes):

        Data_Missing = Gen.GetSurveyAndEventData(min_flux = min_flux)

        I_Missing = Inference(Data_Missing, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type="perfect")

        H_0_sample = I_Missing.H_0_Prob()
        H_0_samples[str(Universe)+"_Missing_"+min_flux_pct[i]] = H_0_sample





#Automated Labelling

def find_file_num(name):
    path = os.getcwd()
    results = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                results.append(file)
    if results == []:
        max_num = 0
        return str(max_num)
    nums = []
    for result in results:
        nums.append(int(result.split("_", -1)[-1].split(".")[0]))
    max_num = np.max(nums) + 1
    return str(max_num)

max_num = find_file_num("SampleUniverse_"+str(dimension)+"_"+str(min_flux)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_")

H_0_samples.to_csv("SampleUniverse_"+str(dimension)+"_"+str(min_flux)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+str(time)+"_"+max_num+".csv")
print("Finished: SampleUniverse_"+str(dimension)+"_"+str(min_flux)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+str(time)+"_"+max_num+".csv")


