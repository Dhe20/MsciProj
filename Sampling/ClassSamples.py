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


#events = 50
#characteristic_luminosity_2 = 0.1
#total_luminosity_2 = 10000/3
#rate = 1000
#time = events/(total_luminosity_2*rate*(1.5*32*np.pi/3000))


class Sampler():
    def __init__(self, dimension=3, size=625, event_rate = 1540.0, sample_time = 0.00019378,# 0.00029066,
                luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                cluster_coeff=0, characteristic_luminosity=1, lower_lim=0.1, total_luminosity=3333.333,
                BVM_c = 15, BVM_k = 2, BVM_kappa = 200, event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                resolution=10, plot_contours=False, alpha = 0.3, beta=-1.5, resolution_H_0 = 200, 
                H_0_Min = 50, H_0_Max = 100, universe_count = 200, survey_type = "perfect", investigated_characteristic='0', investigated_value=0):

        # Event generator params - to use super __init__ have to change defaults in EventGenerator to be useful
        self.dimension = dimension
        self.size = size
        self.event_rate = event_rate
        self.sample_time = sample_time
        self.luminosity_gen_type = luminosity_gen_type
        self.coord_gen_type = coord_gen_type
        self.cluster_coeff = cluster_coeff
        self.characteristic_luminosity = characteristic_luminosity
        self.lower_lim = lower_lim
        self.total_luminosity = total_luminosity
        self.alpha = alpha
        self.beta = beta
        self.BVM_c = BVM_c
        self.BVM_k = BVM_k
        self.BVM_kappa = BVM_kappa
        self.event_distribution = event_distribution
        self.noise_distribution = noise_distribution
        self.redshift_noise_sigma = redshift_noise_sigma
        self.resolution = resolution
        self.plot_contours = plot_contours
                          
        self.universe_count = universe_count
        self.H_0_Min = H_0_Min
        self.H_0_Max = H_0_Max
        self.resolution_H_0 = resolution_H_0
        self.survey_type = survey_type
        self.investigated_characteristic = investigated_characteristic
        self.investigated_value = investigated_value

        self.H_0_samples = pd.DataFrame()

        self.H_0_samples.index = np.linspace(self.H_0_Min, self.H_0_Max, self.resolution_H_0)

        self.events_estimate = self.sample_time * (self.total_luminosity*self.event_rate*(1.5*32*np.pi/3000))

        if self.events_estimate * 10 > self.total_luminosity / self.characteristic_luminosity:
            print("Risk of too many events for galaxy number")

    def Sample(self):
        for Universe in tqdm(range(self.universe_count)):
            Gen = EventGenerator(dimension=self.dimension, size=self.size, event_rate = self.event_rate, sample_time = self.sample_time,
                                luminosity_gen_type=self.luminosity_gen_type, coord_gen_type=self.coord_gen_type,
                                cluster_coeff=self.cluster_coeff, characteristic_luminosity=self.characteristic_luminosity, lower_lim=self.lower_lim, total_luminosity=self.total_luminosity,
                                event_distribution=self.event_distribution, noise_distribution=self.noise_distribution, redshift_noise_sigma=self.redshift_noise_sigma,
                                resolution=self.resolution, plot_contours=self.plot_contours, alpha = self.alpha, beta=self.beta, BVM_c=self.BVM_c, BVM_k=self.BVM_k, BVM_kappa=self.BVM_kappa, seed = Universe)
            print("# of detected events: " + str(Gen.detected_event_count))
            Data = Gen.GetSurveyAndEventData()
            I = Inference(Data, H_0_Min=self.H_0_Min, H_0_Max=self.H_0_Max, resolution_H_0=self.resolution_H_0, survey_type = self.survey_type)
            H_0_sample = I.H_0_Prob()
            self.H_0_samples[Universe] = H_0_sample

        self.max_num = self.find_file_num("PosteriorData/SampleUniverse_"+str(self.investigated_characteristic)+"_"+str(self.investigated_value)+"_")
        self.H_0_samples.to_csv("PosteriorData/SampleUniverse_"+str(self.investigated_characteristic)+"_"+str(self.investigated_value)+"_"+self.max_num+".csv")
        print("Finished: SampleUniverse_"+str(self.investigated_characteristic)+"_"+str(self.investigated_value)+"_"+self.max_num+".csv")


    #Automated Labelling

    def find_file_num(self, name):
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


# %%
