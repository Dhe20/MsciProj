import pandas as pd

from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


characteristic_luminosity= 0.1
total_luminosity= 100
rate = 10**8
sample_time = 1650/(rate*total_luminosity)
iterations = 1
dists = np.zeros((iterations, 200))
for i in tqdm(range(0,iterations)):
    Gen = EventGenerator(
        dimension = 3, luminosity_gen_type = "Full-Schechter",
         coord_gen_type = "Random",
         cluster_coeff = 0, total_luminosity = total_luminosity, size = 625,
         alpha = .3, beta=-1.5, characteristic_luminosity = characteristic_luminosity, min_lum = 0,
         max_lum = 1, lower_lim=1, event_rate = rate, sample_time = sample_time,
         event_distribution = "Proportional",
         noise_distribution = "BVMF_eff", contour_type = "BVM",
         noise_std = 36, resolution = 400, BVM_c = 15, H_0 = 70,
         BVM_k = 2, BVM_kappa = 200, redshift_noise_sigma = 0,
         plot_contours = False, seed = i, event_count_type = "Poisson"
                 )

    Data = Gen.GetSurveyAndEventData()
    I = Inference(Data, gamma = True, gaussian = False, resolution_H_0=200, gamma_known=False)
    I.H_0_Prob()
    dists[i] = I.gamma_marginalised
    plt.plot(I.H_0_range, dists[i]-1)
