from Components.EventGenerator import EventGenerator
import numpy as np
from Sampling.ClassSamples import Sampler
from scipy.integrate import quad
from scipy import interpolate
from tqdm import tqdm



def calc_localisation_volume(BVM_c, BVM_k, BVM_kappa, total_events_required = 100, confidence_pct = .9):


    total_luminosity = 1000
    event_rate = 10 ** 6
    wanted_det_events = 10
    d_ratio = 0.4

    integral, err = quad(lambda x: 3 * (1 - 1 / ((1 + (d_ratio / x) ** BVM_c) ** BVM_k)) * (x ** 2), 0, 1)
    req_time = wanted_det_events / (total_luminosity * event_rate * integral * np.pi / 6)

    number_of_universes = total_events_required // wanted_det_events + 1

    Volumes = []
    PctVolumes = []
    Lengths = []
    total_events_considered = 0

    for seed in tqdm(range(number_of_universes)):
        if total_events_considered > total_events_required:
            continue
        Gen = EventGenerator(dimension = 3, size = 625, resolution = 100, BVM_kappa=BVM_kappa, BVM_k=BVM_k, BVM_c=BVM_c,
                              luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                              cluster_coeff=5, characteristic_luminosity=1, total_luminosity=total_luminosity, sample_time=req_time, event_rate=event_rate,
                              event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0.0, plot_contours=True, seed = seed)
        for i, PDF in enumerate(Gen.BH_detected_meshgrid):
            X, Y, Z = Gen.BH_contour_meshgrid
            n = 200
            PDF = PDF / PDF.sum()
            t = np.linspace(0, PDF.max(), n)
            integral = (((PDF >= t[:, None, None, None]) * PDF).sum(axis=(1, 2, 3)))
            f = interpolate.interp1d(integral, t)
            pct_contour = f(np.array([confidence_pct]))
            volume_index = np.where(PDF > pct_contour)

            num_points = len(volume_index[0])

            # Assuming uniform spacing delta in all dimensions
            delta = np.abs(X[1,0,0] - X[0,0,0])  # The spacing between points in the meshgrid

            # Volume of each small cube
            cube_volume = delta ** 3

            # Total volume occupied by the region where PDF > pct_contour
            total_volume = (num_points * cube_volume)

            #Log Data
            PctVolumes.append(total_volume * 100 / (2 * Gen.size) ** 3)
            Volumes.append(total_volume)
            Lengths.append(((3 / 4) * total_volume) ** (1 / 3))
        total_events_considered += Gen.detected_event_count

    print("Avg Total volume percentage occupied:", np.mean(PctVolumes), "%")
    print("Avg Total volume occupied:", np.mean(Volumes), r"$Mpc^{3}$")
    print("Avg Length Scale:", np.mean(Lengths), r"$Mpc$")
    print("Total Events Used", total_events_considered)

    return np.mean(Volumes)

def calc_centroid_volume(centroid_sigma, gen_size):
    r = 1.645*(centroid_sigma*gen_size)
    V = (4/3) * np.pi * r**3
    print("Centroid 90% Volume occupied:", V, r"$Mpc^{3}$")
    return V

total_events_required = 200

#
# BVM_c = 15
# BVM_k = 2
# BVM_kappa = 200
# print("Small")
# volume_small = calc_localisation_volume(BVM_c = BVM_c, BVM_k = BVM_k, BVM_kappa=BVM_kappa, total_events_required = total_events_required)
#
#
# BVM_c = 8.827
# BVM_k = 2
# BVM_kappa = 20
# print("Medium")
# volume_medium = calc_localisation_volume(BVM_c = BVM_c, BVM_k = BVM_k, BVM_kappa=BVM_kappa, total_events_required = total_events_required)
#
# BVM_c = 3.409
# BVM_k = 2
# BVM_kappa = 20
# print("Big")
# volume_big = calc_localisation_volume(BVM_c = BVM_c, BVM_k = BVM_k, BVM_kappa=BVM_kappa, total_events_required = total_events_required)


