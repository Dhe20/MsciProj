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

#%%
resolution_H_0 = 200
H_0_Min = 50
H_0_Max = 100
universe_count = 100
H_0_samples = pd.DataFrame()

events = 50
characteristic_Luminosity = 0.1
total_luminosity = 100
rate = 1000

time = events/(total_luminosity*rate*(1.5*32*np.pi/3000))

size = 100
dimension = 3

H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

means = []

if events * 10 > total_luminosity / characteristic_Luminosity:
    print("Risk of too many events for galaxy number")

for Universe in tqdm(range(universe_count)):
    Gen = EventGenerator(dimension=dimension, size=size, event_rate = rate, sample_time = time,
                         luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                         cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity, lower_lim=0.1, total_luminosity=total_luminosity,
                         event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                         resolution=10, plot_contours=False, alpha = 0.3, beta=-1.5, seed = Universe)
    # print("# of detected events: " + str(Gen.detected_event_count))
    Data = Gen.GetSurveyAndEventData()
    I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type = "perfect")
    H_0_sample = I.H_0_Prob()
    H_0_samples[Universe] = H_0_sample
    means.append(I.get_mean())

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

max_num = find_file_num("SampleUniverse_"+str(dimension)+"_"+str(rate)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_")

H_0_samples.to_csv("SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+max_num+".csv")
print("Finished: SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+max_num+".csv")

import matplotlib.pyplot as plt
plt.hist(means)


# %%
