import pandas as pd
from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from Visualising.Inference_GUI_2d import InferenceGUI
import pickle


def GetPhis(detected_coords):
    Phis = np.arctan2(detected_coords[:, 1], detected_coords[:, 0])
    return Phis

def Phis2Coords(Phis, Rs):
    Points = np.zeros((len(Phis), 2))
    Points[:, 0] = Rs * np.cos(Phis)
    Points[:, 1] = Rs * np.sin(Phis)
    return Points

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=100,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 42)
Data = Gen.GetSurveyAndEventData()


print(Gen.detected_event_count)

H_0_samples = 400
H_0_min = 50
H_0_max = 100

I = Inference(Data, gamma = True, vectorised = True, event_distribution_inf='Proportional', gauss=False, p_det=True,
                 survey_type='perfect', resolution_H_0=H_0_samples, H_0_Min = H_0_min, H_0_Max = H_0_max, gamma_known = False, gauss_type = "Cartesian")
I.H_0_Prob()




H_0_s = np.linspace(H_0_min,H_0_max, H_0_samples)
coord_arr = np.zeros((2*H_0_samples, len(Gen.detected_coords)))

for i, H_0 in enumerate(H_0_s):
    Rs = Gen.c*Gen.detected_redshifts/H_0
    Phis = GetPhis(Gen.detected_coords)
    Coords = Phis2Coords(Phis, Rs)
    coord_arr[2*i]= Coords[:, 0]
    coord_arr[2*i+1,:]= Coords[:, 1]
    np.savetxt("memorised_coords.csv", coord_arr, delimiter=",")

with open('Generator.pickle', 'wb') as pickle_file:
    pickle.dump(Gen, pickle_file)

with open('Inference.pickle', 'wb') as pickle_file:
    pickle.dump(I, pickle_file)



