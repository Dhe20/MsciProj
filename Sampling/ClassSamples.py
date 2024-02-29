#%%

import sys
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
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
    def __init__(self, dimension=3, cube=True, size=625, d_ratio=0.4, event_rate = 1540.0, sample_time = 0.00019378,# 0.00029066,
                wanted_det_events = 50, specify_event_number = False, luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                cluster_coeff=0, characteristic_luminosity=1, lower_lim=0.1, total_luminosity=3333.333,
                BVM_c = 15, BVM_k = 2, BVM_kappa = 200, event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0, noise_sigma=5,
                resolution=10, plot_contours=False, alpha = 0.3, beta=-1.5, min_flux=0, survey_incompleteness=0, completeness_type='cut_lim', resolution_H_0 = 200, 
                H_0_Min = 50, H_0_Max = 100, universe_count = 200, survey_type = "perfect", gamma=True, gauss=False, p_det=False, event_distribution_inf='Proportioanl', poster=False,
                investigated_characteristic='0', investigated_value=0):

        # Event generator params - to use super __init__ have to change defaults in EventGenerator to be useful
        self.dimension = dimension
        self.cube = cube
        self.size = size
        self.d_ratio = d_ratio
        self.event_rate = event_rate
        self.sample_time = sample_time
        self.wanted_det_events = wanted_det_events
        self.specify_event_number = specify_event_number

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
        self.noise_sigma = noise_sigma
        self.resolution = resolution
        self.plot_contours = plot_contours
        
        self.universe_count = universe_count
        self.H_0_Min = H_0_Min
        self.H_0_Max = H_0_Max
        self.resolution_H_0 = resolution_H_0
        self.survey_type = survey_type
        self.gamma = gamma
        self.gauss = gauss
        self.p_det = p_det
        self.poster = poster

        self.event_distribution_inf = event_distribution_inf

        self.min_flux = min_flux
        self.survey_incompleteness = survey_incompleteness
        self.completeness_type = completeness_type

        self.investigated_characteristic = investigated_characteristic
        self.investigated_value = investigated_value

        if self.specify_event_number:
            self.sample_time = self.factor()

        self.H_0_samples = pd.DataFrame()

        self.H_0_samples.index = np.linspace(self.H_0_Min, self.H_0_Max, self.resolution_H_0)

        self.events_estimate = self.sample_time * (self.total_luminosity*self.event_rate*(1.5*32*np.pi/3000))

        if self.events_estimate * 10 > self.total_luminosity / self.characteristic_luminosity:
            print("Risk of too many events for galaxy number")

    #def p_I(self, x):
    #    return 3 * (1 - 1/((1 + (self.d_ratio/x)**self.BVM_c)**self.BVM_k)) * (x**2)

    def factor(self):
        # radial gauss
        if self.gauss:
            integral, err = quad(lambda x: 3 * 0.5 * (1 + erf((self.size * self.d_ratio - self.size * x) / (np.sqrt(2) * self.noise_sigma ))) * (x**2) , 0, 1)
        else:
            integral, err = quad(lambda x: 3 * (1 - 1/((1 + (self.d_ratio/x)**self.BVM_c)**self.BVM_k)) * (x**2) , 0, 1)
        req_time = self.wanted_det_events/(self.total_luminosity * self.event_rate * integral * np.pi/6)
        return req_time

    def Sample(self):
        det_event_counts = []
        self.survey_percentage = []
        for Universe in range(self.universe_count):
            Gen = EventGenerator(dimension=self.dimension, cube=self.cube, size=self.size, event_rate = self.event_rate, sample_time = self.sample_time,
                                luminosity_gen_type=self.luminosity_gen_type, coord_gen_type=self.coord_gen_type,
                                cluster_coeff=self.cluster_coeff, characteristic_luminosity=self.characteristic_luminosity, lower_lim=self.lower_lim, total_luminosity=self.total_luminosity,
                                event_distribution=self.event_distribution, noise_distribution=self.noise_distribution, redshift_noise_sigma=self.redshift_noise_sigma, noise_std=self.noise_sigma,
                                resolution=self.resolution, plot_contours=self.plot_contours, alpha = self.alpha, beta=self.beta, BVM_c=self.BVM_c, BVM_k=self.BVM_k, BVM_kappa=self.BVM_kappa, seed=Universe)
            det_event_counts.append(Gen.detected_event_count)
            print("# of detected events: " + str(Gen.detected_event_count))
            if Universe==0:    
                print("# of galaxies: " + str(len(Gen.detected_luminosities)))
            Data = Gen.GetSurveyAndEventData(min_flux = self.min_flux, survey_incompleteness = self.survey_incompleteness, completeness_type = self.completeness_type)
            percentage = len(Data.detected_galaxy_indices)/len(Gen.detected_luminosities)
            self.survey_percentage.append(percentage)
            I = Inference(Data, H_0_Min=self.H_0_Min, H_0_Max=self.H_0_Max, resolution_H_0=self.resolution_H_0, survey_type = self.survey_type, gamma = self.gamma, 
                          event_distribution_inf = self.event_distribution_inf, gauss = self.gauss, p_det = self.p_det, poster = self.poster)
            H_0_sample = I.H_0_Prob()
            if not self.poster:    
                self.H_0_samples[Universe] = H_0_sample
        if self.poster:
            for i in range(det_event_counts[0]):
                self.H_0_samples[str(i)] = H_0_sample[i,:]

        #self.burr_i = I.burr_full
        #self.full = I.full
        if self.p_det:    
            self.P_det_total = I.P_det_total
        print(I.inference_method_name)
        self.max_num = self.find_file_num("PosteriorData/SampleUniverse_"+str(self.investigated_characteristic)+"_"+str(self.investigated_value)+"_")
        self.H_0_samples.to_csv("PosteriorData/SampleUniverse_"+str(self.investigated_characteristic)+"_"+str(self.investigated_value)+"_"+self.max_num+".csv")
        print('Average # of detected events = {:.2f}'.format(np.mean(det_event_counts)))
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
