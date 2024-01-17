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

L0_R_T = np.arange(150,2350, 100)
times = L0_R_T/(total_luminosity*rate)

for time in tqdm(times):
    event_num_detected = []
    # events = event
    characteristic_Luminosity = 1
    #
    # time = events/(total_luminosity*rate*(1.2*32*np.pi/3000))

    size = 625
    dimension = 3

    H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

    for Universe in range(universe_count):
        Gen = EventGenerator(dimension=dimension, size=size, event_rate = rate, sample_time = time,
                             luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                             cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity, total_luminosity=total_luminosity,
                             event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                             resolution=100, plot_contours=False, alpha = 0.3, seed = Universe)

        Data = Gen.GetSurveyAndEventData()
        I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type = "perfect")
        H_0_sample = I.H_0_Prob()
        H_0_samples[Universe] = H_0_sample
        event_num_detected.append(Gen.detected_event_count)



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

    H_0_samples.to_csv("SampleUniverse_"+str(dimension)+"_"+str(rate)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+str(time)+"_"+str(np.mean(event_num_detected))+"_"+max_num+".csv")
    print("Finished: SampleUniverse_"+str(dimension)+"_"+str(rate)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+str(time)+"_"+str(np.mean(event_num_detected))+"_"+max_num+".csv")


