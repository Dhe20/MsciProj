from Components.EventGenerator import EventGenerator
# from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
from Components.Inference import Inference
import matplotlib.pyplot as plt
import matplotlib


import os
Gen = EventGenerator(dimension = 3, size = 50, resolution = 100,
                      luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
                      cluster_coeff=5, characteristic_luminosity=1, total_luminosity=100, sample_time=0.1, event_rate=10, H_0 = 100,
                      event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0.0, plot_contours=True, seed = 1)
# print("plotting")
print(Gen.detected_event_count)
Gen.plot_universe_and_events()
# Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data, survey_type='perfect')
# plt.plot(Y.H_0_range, Y.H_0_Prob())
# plt.show()