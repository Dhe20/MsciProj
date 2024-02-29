from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab



Gen = EventGenerator(dimension = 3, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=100,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=100, plot_contours=True, seed = 42)
Data = Gen.GetSurveyAndEventData()
print(Gen.detected_event_count)
# print(len(Gen.detected_redshifts))

Universe_Plot = Sliding_Universe_3d(Gen)
Universe_Plot.configure_traits()