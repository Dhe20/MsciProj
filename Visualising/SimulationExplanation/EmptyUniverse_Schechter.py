from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.integrate import quad

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=2500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=False, seed = 21)
Data = Gen.GetSurveyAndEventData()

fig = plt.figure(figsize=(6, 9))
gs = GridSpec(nrows = 2, ncols = 2, height_ratios=[1, 0.5], figure = fig)

ax0 = plt.subplot(gs[0,:])
ax1 = plt.subplot(gs[1,:])
# ax2 = plt.subplot(gs[1,1])
ax2=None

axs = [ax0, ax1, ax2]


ax0.add_patch(plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill=""))
x, y = zip(*Gen.detected_coords)
cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
ax0.add_patch(cutoff)
for _ in range(Gen.n):
    ax0.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="b",)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))


def schechter_function(L, L_star, phi_star, alpha):
    """
    Calculate the Schechter function.

    Parameters:
    - L (array): Array of luminosities
    - L_star (float): Characteristic luminosity
    - phi_star (float): Normalization constant
    - alpha (float): Slope of the power-law component

    Returns:
    - (array): Schechter function evaluated at L
    """
    return (phi_star * (L / L_star) ** alpha * np.exp(-L / L_star) / L_star)

L_star = Gen.L_star
phi_star = (Gen.n)/(4*Gen.size**2)
alpha = Gen.alpha
Ls = np.linspace(np.min(Gen.true_luminosities), np.max(Gen.true_luminosities))

def integrate_schechter(L_min, L_max, L_star, phi_star, alpha):
    """
    Integrate the Schechter function over a range of luminosities.
    """
    result, _ = quad(schechter_function, L_min, L_max, args=(L_star, phi_star, alpha))
    return result


# bins = np.logspace(np.log10(min(Gen.true_luminosities)), np.log10(max(Gen.true_luminosities)), num=20)
bins = np.linspace(min(Gen.true_luminosities), max(Gen.true_luminosities), num=20)
bin_midpoints = bins[:-1]+0.5*(bins[1]-bins[0])
# bin_midpoints = 10**(0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
# Calculate the expected number in each bin
expected_numbers = []
for i in range(len(bins) - 1):

    L_min, L_max = bins[i], bins[i+1]
    if i == 0:
        L_min = 0
    expected_number = integrate_schechter(L_min, L_max, L_star, phi_star, alpha)
    expected_numbers.append(expected_number)

# print(len(Gen.true_luminosities))
# print(Gen.n)
# print(Gen.size)




# Plot the histogram
# plt.hist(luminosities, bins=bin_edges, edgecolor='black')


# hist = ax1.hist(Gen.true_luminosities,bins=bins , density=False)
# hist_heights = hist[0]

# schechter_plot = np.max(hist_heights)/np.max(schechter_Ls)*schechter_Ls

ax1.plot(bin_midpoints, expected_numbers)


plt.show()
