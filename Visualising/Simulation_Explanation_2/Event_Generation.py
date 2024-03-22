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
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()


fig = plt.figure(figsize=(6, 12))
gs = GridSpec(nrows = 3, ncols = 2, height_ratios=[0.5, 1, 0.5], figure = fig)

ax0 = plt.subplot(gs[1,:])
ax1 = plt.subplot(gs[2,0])
ax2 = plt.subplot(gs[2,1])
ax3 = plt.subplot(gs[0,:])
# ax2=None

axs = [ax0, ax1, ax2, ax3]


ax0.add_patch(plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill=""))
x, y = zip(*Gen.detected_coords)
cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
ax0.add_patch(cutoff)
for _ in range(Gen.n):
    ax0.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="k",)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))

for i in range(Gen.detected_event_count):
    mu = Gen.BH_true_coords[i]
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



    X, Y = Gen.BH_contour_meshgrid
    z = Z
    n = 1000
    z = z / z.sum()
    t = np.linspace(0, z.max(), n)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
    if i == 0:
        ax0.contour(X,Y, z, t_contours, colors="magenta", zorder = 2)
        ax0.scatter(Gen.BH_detected_coords[i, 0], Gen.BH_detected_coords[i, 1], s=200, color='black', marker='x', zorder=5)
    else:
        ax0.contour(X, Y, z, t_contours, colors="g", zorder=2)
        ax0.scatter(Gen.BH_detected_coords[i, 0], Gen.BH_detected_coords[i, 1], s=200, color='red', marker='x',
                    zorder=5)


x, y = zip(*Gen.detected_coords)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    if s in Gen.BH_true_luminosities:
        ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="g", zorder = 3))
ax0.set_xticks([])  # Remove x-ticks
ax0.set_yticks([])


x = np.linspace(u_r - 5*np.sqrt(u_r), u_r + 5*np.sqrt(u_r))
pdf = Gen.burr(x, c = Gen.BVM_c, k = Gen.BVM_k, l = u_r)
# Plotting
ax1.plot(x, pdf, color = 'magenta', lw=2)
ax1.set_xlabel('Radial Distance (Mpc)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Burr XII Distribution')
# ax1.legend(loc='upper left')
plt.grid(True)

# Given kappa value
kappa = Gen.BVM_kappa  # Change this value to see how the distribution changes

# Generate angles from 0 to 2*pi
angles = np.linspace(np.pi/2, 2.5, 10000)

# Calculate the PDF of the von Mises distribution for these angles
pdf_values = vonmises.pdf(angles, kappa, loc = u_phi)

# Plotting
ax2.plot(angles, pdf_values, color = 'magenta', lw=2)
ax2.set_xlabel('Angle (radians)')
ax2.set_ylabel('Probability Density')
ax2.set_title('von Mises Distribution')
# ax2.legend(loc='upper right')  # Set legend to be in the top right


lambda_val = Gen.event_rate*Gen.sample_time*Gen.L_0  # Example value; you can change this as needed

# Generate an array of k values (where k is the number of events) around the lambda value
k_values = np.arange(lambda_val-2*np.sqrt(lambda_val), lambda_val+3*np.sqrt(lambda_val))//1  # You might want to adjust the range based on your lambda

# Calculate the PMF for each k value
pmf_values = poisson.pmf(k_values, lambda_val)


ax3.stem(k_values, pmf_values, basefmt=" ", linefmt="dodgerblue")
ax3.stem(Gen.total_event_count, pmf_values[np.where(k_values==Gen.total_event_count)[0]],
         basefmt=" ", linefmt="crimson", label = r"$N_{events} = $" + str(Gen.total_event_count))
# plt.title('Poisson Distribution (λ = {})'.format(lambda_val))
rectangle_patch = mpatches.Rectangle((min(k_values), min(pmf_values)), width=0, height=0, color='green', label=r'$N_{exp}$ = '+str(round(lambda_val,1)), alpha=0)
ax3.add_patch(rectangle_patch)

ax3.legend(loc='upper right')

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//Event_Generation.svg'


plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)


plt.show()


