from math import isclose; import matplotlib.pyplot as plt
from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()

I = Inference(Data, gamma = True, vectorised = True, event_distribution_inf='Proportional', gauss=False, p_det=True,
                 survey_type='perfect', resolution_H_0=100, H_0_Min = 50, H_0_Max = 100, gamma_known = False, gauss_type = "Cartesian")
I.H_0_Prob()
print(Gen.detected_event_count)
fig, main_ax = Gen.plot_universe_and_events(show = False)

xlim=[-110, 15]
ylim=[130, 310]

x, y, height = 0.01, 0.5, 0.49
width = height*(xlim[-1]-xlim[0])/(ylim[-1]-ylim[0])

inset_ax = main_ax.inset_axes(
   [x, y, width, height],  # [x, y, width, height] w.r.t. axes
    xlim=xlim, ylim=ylim, # sets viewport & tells relation to main axes
    xticklabels=[], yticklabels=[]
)

coord = np.array([-30, 288])
index = np.argmin(np.sum(np.square(Gen.detected_coords - coord),axis = 1))
true_coord = Gen.detected_coords[index]

x, y = zip(*Gen.detected_coords)
cutoff = plt.Circle((0, 0), Gen.max_D, color='w', ls="--", fill="")
inset_ax.add_patch(cutoff)
for _ in range(Gen.n):
    inset_ax.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="b",)
for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
    if i == index:
        neighbouring_galaxy = plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="white", zorder = 3)
    inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="white", zorder = 3))
inset_ax.scatter(0,0, s=Gen.size/1.25, c = "w", marker = "x")
inset_ax.xaxis.set_tick_params(labelbottom=False)
inset_ax.yaxis.set_tick_params(labelleft=False)
inset_ax.set_xticks([])
inset_ax.set_yticks([])
plt.tight_layout()


for (xhat, yhat, s) in zip(*zip(*Gen.BH_detected_coords), Gen.BH_detected_luminosities):
    inset_ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=Gen.L_star, color="red", zorder = 4))
    detected_patch = plt.Circle(xy=(xhat, yhat), radius=Gen.L_star, color="red", zorder=4)



for i, Z in enumerate(Gen.BH_detected_meshgrid):
    X, Y = Gen.BH_contour_meshgrid
    z = Z
    n = 1000
    z = z / z.sum()
    t = np.linspace(0, z.max(), n)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
    inset_ax.contour(X,Y, z, t_contours, colors="red", zorder = 2)

x, y = zip(*Gen.BH_true_coords)
for (x, y, s) in zip(x, y, Gen.BH_true_luminosities):
    inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="cyan", zorder = 4))
    true_patch = plt.Circle(xy=(x, y), radius=s, color="cyan", zorder=4)

inset_ax.annotate(r'$\boldsymbol{Detected \; Location}$' "\n" r'$\boldsymbol{of \; GW \; Signal}$',xy=(Gen.BH_detected_coords[-1]), xycoords='data',xytext=(30,260),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="red",
                    ec="none",
                    patchB=detected_patch,
                    connectionstyle="arc3,rad=0.2"), size = 16)

inset_ax.annotate(r'$\boldsymbol{True \; Host \; Galaxy}$' "\n" r'$Unknown \; to \; Observer$',xy=(Gen.BH_true_coords[-1]), xycoords='data',xytext=(-108, 140),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="turquoise",
                    ec="none",
                    patchB=true_patch,
                    connectionstyle="arc3,rad=-0.2"), size = 16)


inset_ax.annotate(r'$\boldsymbol{Potential \; Host}$' "\n" r'$\boldsymbol{Galaxy}$',xy=(true_coord), xycoords='data',xytext=(20,290),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="white",
                    patchB = neighbouring_galaxy,
                    ec="none",
                    connectionstyle="arc3,rad=0.1"), size = 16)

inset_ax.annotate(r'$\boldsymbol{1, 2, 3 \,\sigma \, Contours}$' "\n" r'$\boldsymbol{of \; True \; Host \; Location}$',xy=(11, 220), xycoords='data',xytext=(40, 230),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="red",
                    ec="none",
                    connectionstyle="arc3,rad=+0.3"), size = 16)

inset_ax.annotate("\n",xy=(-4, 220), xycoords='data',xytext=(40, 230),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="red",
                    ec="none",
                    connectionstyle="arc3,rad=+0.3"), size = 16)

inset_ax.annotate("\n",xy=(-21, 220), xycoords='data',xytext=(40, 230),
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                    fc="red",
                    ec="none",
                    connectionstyle="arc3,rad=+0.4"), size = 16)





main_ax.indicate_inset_zoom(inset_ax, edgecolor="white")

for axis in ['top','bottom','left','right']:
    main_ax.spines[axis].set_linewidth(3)

# plt.savefig("2DAnnotatedUniverse.png", dpi = 1500)

# careful! warn if aspect ratio of inset axes doesn't match main axes
# if not isclose(inset_ax._get_aspect_ratio(), main_ax._get_aspect_ratio()):
#     print("chosen inset x/ylim & width/height skew aspect w.r.t. main axes!")