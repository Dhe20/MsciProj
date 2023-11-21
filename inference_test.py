#%%
import numpy as np
from EventGenerator import EventGenerator
from Inference import Inference
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%

sim_n = 2
means = []
stds = []
posteriors = []

for i in tqdm(range(sim_n)):
    Gen = EventGenerator(dimension = 2, size = 50, event_count=50,
                        luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Clustered",
                        cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=100,
                        event_distribution="Proportional", noise_distribution = "BVM", contour_type = "BVM", redshift_noise_sigma = 0.001,
                        resolution=500)

    #Gen.plot_universe_and_events()
    print(i)
    Data = Gen.GetSurveyAndEventData()
    Y = Inference(Data, survey_type='perfect')
    print(i)
    pdf = Y.H_0_posterior()
    mean = np.sum(pdf[0]*pdf[1])
    std = np.sqrt(np.sum(pdf[1]*(pdf[0]-mean)**2))
    means.append(mean)
    stds.append(std)
    posteriors.append(pdf)

cm = plt.cm.plasma(np.linspace(0, 1, sim_n))
for i in range(sim_n):
        plt.plot(posteriors[i][0],posteriors[i][1], color=cm[i])
        plt.axvline(x=70, c='white', ls='--')
plt.show()

#%%

plt.hist(means)
plt.show()
plt.hist(stds)
plt.show()