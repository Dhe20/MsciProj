from Components.EventGenerator import EventGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Gen = EventGenerator(dimension=3, size=625, event_rate = 10, sample_time = 0,
                             luminosity_gen_type="Full-Schechter", coord_gen_type="Random",
                             cluster_coeff=5, characteristic_luminosity=1, total_luminosity=10000,
                             event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0,
                             resolution=100, plot_contours=False, alpha = 0.3, seed = 0)

bin_count, bin_edges = np.histogram(Gen.fluxes, bins=np.logspace(-14,2,5000))

bin_pdf = bin_count/np.sum(bin_count)
bin_cdf = np.cumsum(bin_pdf)
pcts = [0.1, 0.5, 1, 5, 10, 50, 99]
flux_limits = {}

for pct in pcts:
    flux_limit = bin_edges[np.where(bin_cdf > pct/100)[0][0]+1]
    flux_limits[str(pct)] = (flux_limit)
print(flux_limits)

Data_True = Gen.GetSurveyAndEventData()
Data_Missing = Gen.GetSurveyAndEventData(min_flux = flux_limits['10'])

print(1-len(Data_Missing.fluxes)/len(Data_True.fluxes))
