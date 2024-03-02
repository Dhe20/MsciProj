from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from Visualising.Inference_GUI_2d import InferenceGUI


Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()

I = Inference(Data, gamma = True, vectorised = True, event_distribution_inf='Proportional', gauss=False, p_det=True,
                 survey_type='perfect', resolution_H_0=100, H_0_Min = 50, H_0_Max = 100, gamma_known = False, gauss_type = "Cartesian")
I.H_0_Prob()
print(Gen.detected_event_count)
Gen.plot_universe_and_events()


