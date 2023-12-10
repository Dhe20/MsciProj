import pandas as pd
import numpy as np
from Components.Inference import Inference
from Components.EventGenerator import EventGenerator
import random

dimension = 3
resolution_H_0 = 200
H_0_Min = 50
H_0_Max = 100
universe_count = 10
H_0_samples = pd.DataFrame()

events = 50
characteristic_Luminosity = 0.1
total_luminosity = 50
size = 100

H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

if events * 10 > total_luminosity / characteristic_Luminosity:
    print("Risk of too many events for galaxy number")

for Universe in range(universe_count):
    Gen = EventGenerator(dimension=dimension, size=size, event_count=events,
                         luminosity_gen_type="Cut-Schechter", coord_gen_type="Random",
                         cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity, total_luminosity=total_luminosity,
                         event_distribution="Proportional", contour_type="BVM", redshift_noise_sigma=0,
                         resolution=100, plot_contours=False, alpha = 0.3)

    Data = Gen.GetSurveyAndEventData()
    I = Inference(Data, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0)
    H_0_sample = I.H_0_Prob()
    H_0_samples[Universe] = H_0_sample

RandNum = str(random.randint(1,10000))

H_0_samples.to_csv("SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+RandNum+".csv")
print("Finished: SampleUniverse_"+str(dimension)+"_"+str(events)+"_"+str(characteristic_Luminosity)+"_"+str(total_luminosity)+"_"+RandNum+".csv")


