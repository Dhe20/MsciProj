
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.integrate import quad
from scipy.stats import poisson
plt.style.use('default')
from scipy import interpolate
from scipy.stats import burr12

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=250,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=True, seed = 23)
Data = Gen.GetSurveyAndEventData()

fig = plt.figure(figsize=(9, 6))
gs = GridSpec(nrows = 1, ncols = 3, figure = fig)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])
# ax2=None

axs = [ax1, ax2, ax3]


mu = Gen.BH_true_coords[0]
u_x = mu[0]
u_y = mu[1]
s_x = Gen.noise_sigma
s_y = Gen.noise_sigma
X = Gen.BH_contour_meshgrid[0]
Y = Gen.BH_contour_meshgrid[1]
r = np.sqrt((X) ** 2 + (Y) ** 2)
phi = np.arctan2(Y, X)
u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2)
u_phi = np.arctan2(u_y, u_x)

k = Gen.BVM_k
c = Gen.BVM_c
kappa = Gen.BVM_kappa

angular = Gen.von_misses(phi, u_phi, kappa)
radial = Gen.burr(r, c, k, u_r)
Z = r * angular * radial

x = np.linspace(u_r - 5*np.sqrt(u_r), u_r + 5*np.sqrt(u_r))
pdf = Gen.burr(x, c = Gen.BVM_c, k = Gen.BVM_k, l = u_r)
# Plotting
ax1.plot(x, pdf, 'r-', lw=2)
ax1.set_xlabel('Radial Distance (Mpc)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Burr XII Distribution')
# ax1.legend(loc='upper left')
plt.grid(True)

# Given kappa value
kappa = Gen.BVM_kappa  # Change this value to see how the distribution changes

# Generate angles from 0 to 2*pi
angles = np.linspace(np.pi/2, np.pi, 10000)

# Calculate the PDF of the von Mises distribution for these angles
pdf_values = vonmises.pdf(angles, kappa, loc = u_phi)

# Plotting
ax2.plot(angles, pdf_values)
ax2.set_xlabel('Angle (radians)')
ax2.set_ylabel('Probability Density')
ax2.set_title('von Mises Distribution')
# ax2.legend(loc='upper right')  # Set legend to be in the top right

spine_width = 2
ax1.spines['top'].set_linewidth(spine_width)
ax1.spines['right'].set_linewidth(spine_width)
ax1.spines['bottom'].set_linewidth(spine_width)
ax1.spines['left'].set_linewidth(spine_width)
# ax1.spines['top'].set_edgecolor('green')
# ax1.spines['right'].set_edgecolor('green')
# ax1.spines['bottom'].set_edgecolor('green')
# ax1.spines['left'].set_edgecolor('green')
ax2.spines['top'].set_linewidth(spine_width)
ax2.spines['right'].set_linewidth(spine_width)
ax2.spines['bottom'].set_linewidth(spine_width)
ax2.spines['left'].set_linewidth(spine_width)
# ax2.spines['top'].set_edgecolor('green')
# ax2.spines['right'].set_edgecolor('green')
# ax2.spines['bottom'].set_edgecolor('green')
# ax2.spines['left'].set_edgecolor('green')
ax3.spines['top'].set_linewidth(spine_width)
ax3.spines['right'].set_linewidth(spine_width)
ax3.spines['bottom'].set_linewidth(spine_width)
ax3.spines['left'].set_linewidth(spine_width)

plt.show()

