from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.integrate import quad
from scipy.stats import poisson
plt.style.use('default')
from scipy import interpolate

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()

# fig = plt.figure(figsize=(9, 9))

ax0 = plt.subplot()
ax1=None
ax2=None

axs = [ax0, ax1, ax2]


# ax0.add_patch(plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill=""))
x, y = zip(*Gen.detected_coords)
# cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
# ax0.add_patch(cutoff)
for _ in range(Gen.n):
    ax0.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="k",)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))


ax0.set_xticks([])  # Remove x-ticks
ax0.set_yticks([])

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//Universe_Generation.svg'


plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)


plt.show()