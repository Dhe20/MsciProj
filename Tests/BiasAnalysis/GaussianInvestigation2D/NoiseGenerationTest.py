import pandas as pd
#
from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from Visualising.Inference_GUI_2d import InferenceGUI

iterations = 1

means = []
df = pd.DataFrame()
characteristic_luminosity= 1
total_luminosity= 1000
rate = 10**6
sample_time = 10*1650/(rate*total_luminosity)
iterations = 1
for i in tqdm(range(iterations)):
    Gen = EventGenerator(
     dimension = 2, luminosity_gen_type = "Full-Schechter",
      coord_gen_type = "Random",
      cluster_coeff = 0, total_luminosity = total_luminosity, size = 625,
      alpha = .3, beta=-1.5, characteristic_luminosity = characteristic_luminosity, min_lum = 0,
      max_lum = 1, lower_lim=1, event_rate = rate, sample_time = sample_time,
      event_distribution = "Random",
      noise_distribution = "GVMF_eff", contour_type = "GVMF",
      noise_std = 10, resolution = 400, BVM_c = 15, H_0 = 70,
      BVM_k = 2, BVM_kappa = 200, redshift_noise_sigma = 0,
      plot_contours = False, seed = i, event_count_type = "Poisson"
              )
    rdiff = np.linalg.norm(Gen.BH_true_coords, axis = 1) - np.linalg.norm(Gen.BH_detected_coords, axis = 1)
    thetadiff = np.arctan2(Gen.BH_true_coords[:,1], Gen.BH_true_coords[:,0]) - np.arctan2(Gen.BH_detected_coords[:,1], Gen.BH_detected_coords[:,0])
    thetadiff[np.where(thetadiff<-6)] +=2*np.pi
    thetadiff[np.where(thetadiff > 6)] -= 2 * np.pi

    print("Radial Distribution: Mean", np.mean(rdiff), "Std:", np.std(rdiff), "Exp. Std", Gen.noise_sigma)
    # plt.hist(rdiff, bins = 30)
    print("Von Mises Distribution: Mean", np.mean(thetadiff), "Std:", np.std(thetadiff), "Exp. Std", (1/Gen.BVM_kappa)**0.5)
    # plt.hist(thetadiff,bins=30)