from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
# from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
# from Visualising.Inference_GUI_2d import InferenceGUI
import matplotlib.pyplot as plt


# while OutsideBox:
Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.4*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=.5, total_luminosity=100,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=False, seed = 10)


print(Gen.detected_event_count)
investigated_values = np.array([1.0]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
# print(np.mean(Gen.fluxes))
print(len(Gen.fluxes[np.where(Gen.fluxes<investigated_values)[0]])/len(Gen.fluxes))

Data = Gen.GetSurveyAndEventData(min_flux=investigated_values[0])
Data_Whole = Gen.GetSurveyAndEventData(min_flux=0)

I = Inference(Data, H_0_Min=50, H_0_Max = 100, resolution_H_0 = 200)
I_Whole = Inference(Data_Whole, H_0_Min=50, H_0_Max = 100, resolution_H_0 = 200)

N1 = []
N1_Whole = []

for H_0 in I.H_0_range:
    N1.append(I.calc_N1(H_0)*Gen.event_rate)
    N1_Whole.append(I_Whole.calc_N1(H_0)*Gen.event_rate)

plt.plot(I.H_0_range, N1, label = "")
plt.plot(I.H_0_range, N1_Whole)
plt.show()


# print(Gen.detected_event_count)




# print(len(Gen.detected_redshifts))

# Gen.plot_universe_and_events()

