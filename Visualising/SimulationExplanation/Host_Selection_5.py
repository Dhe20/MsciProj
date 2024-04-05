from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.integrate import quad
from scipy.stats import poisson
plt.style.use('default')

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=250,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=False, seed = 23)
Data = Gen.GetSurveyAndEventData()

fig = plt.figure(figsize=(6, 9))
gs = GridSpec(nrows = 2, ncols = 2, height_ratios=[1, 0.5], figure = fig)

ax0 = plt.subplot(gs[0,:])
ax1 = plt.subplot(gs[1,0])
ax2 = plt.subplot(gs[1,1])
# ax2=None

axs = [ax0, ax1, ax2]


ax0.add_patch(plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill=""))
x, y = zip(*Gen.detected_coords)
cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
ax0.add_patch(cutoff)
for _ in range(Gen.n):
    ax0.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="k",)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))


x, y = zip(*Gen.detected_coords)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    if s == Gen.BH_true_luminosities[0]:
        ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="r", zorder = 3))

lambda_val = Gen.event_rate*Gen.sample_time*Gen.L_0  # Example value; you can change this as needed

# Generate an array of k values (where k is the number of events) around the lambda value
k_values = np.arange(lambda_val-2*np.sqrt(lambda_val), lambda_val+3*np.sqrt(lambda_val))//1  # You might want to adjust the range based on your lambda

# Calculate the PMF for each k value
pmf_values = poisson.pmf(k_values, lambda_val)

# Plotting
ax1.stem(k_values, pmf_values, basefmt=" ", linefmt="dodgerblue")
ax1.stem(Gen.total_event_count, pmf_values[np.where(k_values==Gen.total_event_count)[0]],
         basefmt=" ", linefmt="crimson", label = r"$N_{events} = $" + str(Gen.total_event_count))
# plt.title('Poisson Distribution (λ = {})'.format(lambda_val))
rectangle_patch = mpatches.Rectangle((min(k_values), min(pmf_values)), width=0, height=0, color='green', label=r'$N_{exp}$ = '+str(round(lambda_val,1)), alpha=0)
ax1.add_patch(rectangle_patch)

ax1.legend(loc='upper right')

ax1.set_xlabel(r'Number of Generated Events ($N_{events}$)')
ax1.set_ylabel('Probability')
# ax1.set_xticks(k_values[::2])
ax1.grid(axis='y', linestyle='--')
ax1.legend(loc='upper right')
# plt.show()
# ax1.plot(bin_midpoints, expected_numbers)

heights, edges = np.histogram(Gen.true_luminosities, bins = len(Gen.true_luminosities), density = True)

# bin_width = edges[1]-edges[0]
# centers = edges[:-1] + (bin_width)/2
# weights = heights*centers
# ax2.plot(centers, weights, color = 'dodgerblue', ls = "-")
# gal_one = Gen.BH_detected_luminosities[0]
# gal_one_index = np.digitize(gal_one, edges)-1
# print(weights[gal_one_index-1])
# gal_one_bin_center = centers[gal_one_index]
# ax2.plot([centers[gal_one_index] - bin_width/2 ,centers[gal_one_index],centers[gal_one_index] + bin_width/2],
#          [0, weights[gal_one_index],0], color = 'crimson')
# ax2.fill_between(gal_one_index)

ax2.stem(sorted(Gen.true_luminosities), np.array(sorted(Gen.true_luminosities))/Gen.L_0, basefmt=" ", linefmt="dodgerblue")
host_gal = np.where(np.array(sorted(Gen.true_luminosities)) == Gen.BH_true_luminosities[0])[0][0]
ax2.stem(sorted(Gen.true_luminosities)[host_gal],
         np.array(sorted(Gen.true_luminosities))[host_gal]/Gen.L_0, basefmt=" ", linefmt="crimson")
ax2.grid(axis='y', linestyle='--')
# ax2.set_yscale("log")
# ax2.set_xscale("log")

spine_width = 2
ax0.spines['top'].set_linewidth(spine_width)
ax0.spines['right'].set_linewidth(spine_width)
ax0.spines['bottom'].set_linewidth(spine_width)
ax0.spines['left'].set_linewidth(spine_width)
ax0.spines['top'].set_edgecolor('green')
ax0.spines['right'].set_edgecolor('green')
ax0.spines['bottom'].set_edgecolor('green')
ax0.spines['left'].set_edgecolor('green')

plt.show()
