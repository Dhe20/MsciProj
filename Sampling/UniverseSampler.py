#%%
import sys
import pandas as pd
import numpy as np
from Components.Inference import Inference
from Components.EventGenerator import EventGenerator
import random

print('1')
resolution_H_0 = 200
H_0_Min = 50
H_0_Max = 100
universe_count = 50
H_0_samples = pd.DataFrame()

events = 10
characteristic_Luminosity = 0.1
total_luminosity = 100
rate = 50

time = 10*events/(total_luminosity*rate*(4*np.pi/3)*(0.4**3))

size = 100
dimension = 3

H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

if events * 10 > total_luminosity / characteristic_Luminosity:
    print("Risk of too many events for galaxy number")

for Universe in range(universe_count):
    #if Universe == 1:
    #    continue
    print(Universe)
    Gen = EventGenerator(dimension=dimension, size=size, event_rate = rate, sample_time = time,
                         luminosity_gen_type="Cut-Schechter", coord_gen_type="Random",
                         cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity, total_luminosity=total_luminosity,
                         event_distribution="Proportional", contour_type="BVM", redshift_noise_sigma=0,
                         resolution=100, plot_contours=False, alpha = 0.3)

    Data = Gen.GetSurveyAndEventData()
    #if Universe==0:
    I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type = "perfect")
    #if Universe==1:
    #I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0, survey_type = "gamma")
    H_0_sample = I.H_0_Prob()
    H_0_samples[Universe] = H_0_sample

RandNum = str(random.randint(1,10000))

# Gen.plot_universe_and_events()

H_0_samples.to_csv("SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+RandNum+".csv")
print("Finished: SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+RandNum+".csv")



# %%
