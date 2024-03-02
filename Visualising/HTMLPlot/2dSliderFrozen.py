from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from Visualising.Inference_GUI_2d_frozen import InferenceGUI
from tqdm import tqdm


Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()

I = Inference(Data, gamma = True, vectorised = True, event_distribution_inf='Proportional', gauss=False, p_det=True,
                 survey_type='perfect', resolution_H_0=100, H_0_Min = 49.5, H_0_Max = 102, gamma_known = False, gauss_type = "Cartesian")

I.H_0_Prob()

H_0_s = np.arange(50, 101, 0.5)
for H_0 in tqdm(H_0_s):
    InferenceGUI(I, Data, Gen, H_0 = H_0).view()
    if H_0%1 == 0:
        H_0 = round(H_0)
    else:
        H_0 = str(H_0)
        H_0 = H_0[0:2] + "_" + H_0[-1]
    plt.close("all")
    plt.savefig("images/range-slider/"+str(H_0)+".jpg", dpi = 400)
    # plt.close("all")





# print(len(Gen.detected_redshifts))

# Universe_Plot = Sliding_Universe_3d(Gen)
# Universe_Plot.configure_traits()