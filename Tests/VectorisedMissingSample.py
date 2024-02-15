from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
import pandas as pd

resolution_H_0 = 200
H_0_Min = 50
H_0_Max = 100
universe_count = 2
total_luminosity = 1000
rate = 1000

H_0_samples = pd.DataFrame()


# event_num = np.linspace(5,50,10).astype(int)
# times =event_num/(total_luminosity*rate*(1.2*32*np.pi/3000))

L0_R_T = 1487.77872778
times = L0_R_T/(total_luminosity*rate)
time = times
min_fluxes = [5.6305408028428945e-06]
min_flux_pct = [50]

for Universe in range(universe_count):

    event_num_detected = []
    characteristic_Luminosity = 1

    size = 625
    dimension = 3

    H_0_samples.index = np.linspace(H_0_Min, H_0_Max, resolution_H_0)

    for i, min_flux in enumerate(min_fluxes):
        Gen = EventGenerator(dimension=dimension, size=size, event_rate = rate, sample_time = time,
                             luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                             cluster_coeff=5, characteristic_luminosity=characteristic_Luminosity, total_luminosity=total_luminosity,
                             event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                             resolution=100, plot_contours=False, alpha = 0.3, seed = Universe)

        Data_Missing = Gen.GetSurveyAndEventData(min_flux=min_flux)

        I_Missing = Inference(Data_Missing, H_0_Min=H_0_Min, H_0_Max=H_0_Max, resolution_H_0=resolution_H_0,
                              survey_type="perfect")
        H_0_sample = I_Missing.H_0_Prob()
        H_0_samples[str(Universe) + "_Missing_" + str(min_flux_pct[i])] = H_0_sample
