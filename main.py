from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
from Visualising.Inference_GUI_2d import InferenceGUI
import matplotlib.pyplot as plt


# while OutsideBox:
Gen = EventGenerator(dimension = 3, size = 50, sample_time=.1, event_rate=10,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=10/3,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 10)
Data = Gen.GetSurveyAndEventData()
# print(Gen.detected_event_count)
# print(len(Gen.detected_redshifts))



# Gen.plot_universe_and_events()


Y = Inference(Data, H_0_Min=50, H_0_Max = 140, resolution_H_0 = 5)
Y.H_0_Prob()
# for i in range(len(Y.H_0_pdf_single_event)):
#     plt.plot(Y.H_0_range, Y.H_0_pdf_single_event[i])
#     plt.show()
# Y.plot_H_0()
# print(Y.H_0_range[np.argmax(Y.H_0_pdf)])
# Gen.plot_universe_and_events()
# GUI = InferenceGUI(Y, Data, Gen)
# GUI.View()

# my_model = Sliding_Universe_3d(Gen)
# my_model.configure_traits()
#
for elem in Y.H_0_range:
    plt.plot(elem, Y.g_H_0[str(elem)][0], "x")

plt.show()