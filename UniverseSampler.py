import pandas as pd
import numpy as np
from Inference import Inference
from EventGenerator import EventGenerator
import matplotlib.pyplot as plt

H_0_Resolution = 200
H_0_min = 50
H_0_max = 90
universe_count = 2
H_0_samples = pd.DataFrame()

Events = 100
Characteristic_Luminosity = 0.1
Total_Luminosity = 500

H_0_samples.index = np.linspace(H_0_min, H_0_max, H_0_Resolution)

for Universe in range(universe_count):
    Gen = EventGenerator(dimension=3, size=50, event_count=10,
                         luminosity_gen_type="Cut-Schechter", coord_gen_type="Clustered",
                         cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=100,
                         event_distribution="Proportional", contour_type="BVM", redshift_noise_sigma=0,
                         resolution=200, plot_contours=True)

    Data = Gen.GetSurveyAndEventData()

