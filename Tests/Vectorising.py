import numpy as np
from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
from Components.Inference import Inference
import time


Gen = EventGenerator(dimension = 3, size = 50, resolution = 100,
                      luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                      cluster_coeff=5, characteristic_luminosity=1, total_luminosity=100000/3, sample_time=0.00001, event_rate=2000,
                      event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0.0, plot_contours=False, seed = 10)
print(Gen.detected_event_count)
print(len(Gen.detected_luminosities))
H_0_Min = 50
H_0_Max = 100
resolution_H_0=100

H_0_range = np.linspace(H_0_Min, H_0_Max, resolution_H_0)
H_0_increment = H_0_range[1] - H_0_range[0]

SurveyAndEventData = Gen.GetSurveyAndEventData()

start_time = time.perf_counter()

H_0_recip = np.reciprocal(H_0_range)[:, np.newaxis]

Zs = np.tile(SurveyAndEventData.detected_redshifts, (resolution_H_0, 1))

Ds = Zs * H_0_recip

Ds_tile = np.tile(Ds,(SurveyAndEventData.detected_event_count, 1,1))

recip_Ds_tile = np.reciprocal(Ds_tile)

u_r = np.sqrt(np.sum(np.square(SurveyAndEventData.BH_detected_coords), axis=1))[:,np.newaxis, np.newaxis]

omegas = recip_Ds_tile * u_r

burr_term1 = np.power(omegas, SurveyAndEventData.BVM_c - 1)
burr_term2 = np.power(1+np.power(omegas, SurveyAndEventData.BVM_c), - SurveyAndEventData.BVM_k-1)

burr_full = SurveyAndEventData.BVM_k*SurveyAndEventData.BVM_c*recip_Ds_tile*burr_term1*burr_term2

#VMF Calc

start_time_vmf = time.perf_counter()

kappa = SurveyAndEventData.BVM_kappa
vmf_C = kappa/(2*np.pi*(np.exp(kappa) - np.exp(-kappa)))

u_x = SurveyAndEventData.BH_detected_coords[:,0]
u_y = SurveyAndEventData.BH_detected_coords[:,1]
u_z = SurveyAndEventData.BH_detected_coords[:,2]

u_phi = np.arctan2(u_y, u_x)[:, np.newaxis]
u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)[:, np.newaxis]

X = SurveyAndEventData.detected_coords[:,0]
Y = SurveyAndEventData.detected_coords[:,1]
Z = SurveyAndEventData.detected_coords[:,2]
XY = np.sqrt((X) ** 2 + (Y) ** 2)

phi = np.tile(np.arctan2(Y, X), (SurveyAndEventData.detected_event_count, 1))
theta = np.tile(np.arctan2(XY, Z), (SurveyAndEventData.detected_event_count, 1))

sin_u_theta = np.sin(u_theta)
cos_u_theta = np.cos(u_theta)

sin_theta = np.sin(theta)
cos_theta = np.cos(theta)

cos_phi_diff = np.cos(phi-u_phi)

vmf = vmf_C*np.exp(kappa*(sin_theta*sin_u_theta*cos_phi_diff + cos_theta*cos_u_theta))[:, np.newaxis, :]

fluxes = SurveyAndEventData.fluxes
luminosity_term = np.square(Ds)*fluxes

full_expression = burr_full*vmf*luminosity_term
posterior = np.product(np.sum(full_expression, axis = 2), axis = 0)
posterior /= np.sum(posterior) * (H_0_increment)

end_time_vmf = time.perf_counter()

# Calculate elapsed time
elapsed_time_vmf = end_time_vmf - start_time_vmf

print("Elapsed time: ", elapsed_time_vmf)

end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

# Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data, survey_type='perfect', gamma = False)
# posteriorY = Y.H_0_Prob()

# plt.plot(H_0_range, posterior)
# plt.plot(H_0_range, posteriorY)
# plt.plot(H_0_range, posterior-posteriorY)
# plt.yscale("log")
# plt.show()





