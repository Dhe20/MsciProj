from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
from Visualising.Inference_GUI_2d import InferenceGUI

Gen = EventGenerator(dimension = 3, size = 50, event_count=1,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=50,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 42)

# Gen.plot_universe_and_events()

Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data, H_0_Min=40, H_0_Max = 140)
# Y.plot_H_0()
# print(Y.H_0_range[np.argmax(Y.H_0_pdf)])
# Gen.plot_universe_and_events()
#
# GUI = InferenceGUI(Y, Data, Gen)
# GUI.View()

my_model = Sliding_Universe_3d(Gen)
my_model.configure_traits()