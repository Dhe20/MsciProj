import pandas as pd
#
from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
# import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from Visualising.Inference_GUI_2d import InferenceGUI




means = []
df = pd.DataFrame()
characteristic_luminosity= 1
total_luminosity= 1000
rate = 10**6
sample_time = 0.05*1650/(rate*total_luminosity)
Samples = 1
for i in tqdm(range(Samples)):
 Gen = EventGenerator(
     dimension = 3, luminosity_gen_type = "Full-Schechter",
      coord_gen_type = "Random",
      cluster_coeff = 0, total_luminosity = total_luminosity, size = 625,
      alpha = .3, beta=-1.5, characteristic_luminosity = characteristic_luminosity, min_lum = 0,
      max_lum = 1, lower_lim=1, event_rate = rate, sample_time = sample_time,
      event_distribution = "Random",
      noise_distribution = "BVMF_eff", contour_type = "BVM",
      noise_std = 10, resolution = 400, BVM_c = 15, H_0 = 70,
      BVM_k = 2, BVM_kappa = 200, redshift_noise_sigma = 0,
      plot_contours = True, seed = i, event_count_type = "Poisson"
              )
 Data = Gen.GetSurveyAndEventData()
 I = Inference(Data, gamma = True, gaussian = True, resolution_H_0=100, H_0_Min=60, H_0_Max=80)
 I.H_0_Prob()
 # means.append(I.get_mean())

#
#
# GUI = InferenceGUI(I, Data, Gen)
# GUI.view()
# plt.plot(I.H_0_range, I.H_0_pdf)
# plt.show()

# Unknown_Gamma_Means = [70.29401385064756, 71.35538318885692, 70.37956131767052, 70.15184663602679, 69.98882337046003, 70.88071784587756, 70.74618175020146, 71.11067670789416, 70.15192584842794, 71.42452943844037, 70.58493236443657, 70.56814585132794, 70.38724213088669, 70.39681784204198, 71.21338045703642, 70.44954103019313, 70.44538553679092, 70.59412780421805, 70.17810137751057, 70.66171885538232]
# Known_Gamma_Means = [70.29401385064756, 71.35538318885692, 70.37956131767052, 70.15184663602679, 69.98882337046003, 70.88071784587756, 70.74618175020146, 71.11067670789416, 70.15192584842794, 71.42452943844037, 70.58493236443657, 70.56814585132794, 70.38724213088669, 70.39681784204198, 71.21338045703642, 70.44954103019313, 70.44538553679092, 70.59412780421805, 70.17810137751057, 70.66171885538232]

Gen.plot_universe_and_events()
# rs = np.linalg.norm(Gen.BH_detected_coords, axis=1)
# u_x = Data.BH_detected_coords[:, 0]
# u_y = Data.BH_detected_coords[:, 1]
#
#
# # u_phi = np.arctan2(u_y, u_x)
# # print(np.mean(rs))
#
# # print(Gen.detected_redshifts/70)
# # plt.hist(rs, bins = 30)
# # plt.show()
# # plt.hist(u_phi)
# # plt.show()
# # print(rs)
#
# print(Gen.detected_event_count)