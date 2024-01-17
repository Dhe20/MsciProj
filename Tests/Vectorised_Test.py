import numpy as np
from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
from Components.Inference import Inference
import time

start_time = time.perf_counter()


Gen = EventGenerator(dimension = 3, size = 50, resolution = 100,
                      luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                      cluster_coeff=5, characteristic_luminosity=1, total_luminosity=10000/3, sample_time=0.001, event_rate=100,
                      event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0.0, plot_contours=False, seed = 10,
                     noise_distribution = "BVMF_eff")
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

Data = Gen.GetSurveyAndEventData()
Y = Inference(Data, survey_type='perfect', gamma = False)

start_time = time.perf_counter()

posterior = Y.H_0_Prob()

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

Y2 = Inference(Data, survey_type='perfect', gamma = False, vectorised = False)

start_time = time.perf_counter()

posterior2 = Y2.H_0_Prob()

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)



plt.plot(Y.H_0_range, posterior, label = "vectorised")
plt.plot(Y.H_0_range, posterior2, label = "normal", ls = "--")
plt.plot(Y.H_0_range, posterior-posterior2, label = "residue")
plt.legend()
plt.yscale("log")
plt.show()