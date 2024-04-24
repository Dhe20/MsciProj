from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from Visualising.Inference_GUI_2d_frozen import InferenceGUI
from tqdm import tqdm
from math import isclose; import matplotlib.pyplot as plt
from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from Visualising.Sliding_Universe_3d import Sliding_Universe_3d
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
plt.style.use("default")

indices = [0,1,3,4,5,6,8]

for seed in [40]:

    Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.007*10**(-2), event_rate=10**3,
                         luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                         cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                         event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                             resolution=400, plot_contours=True, seed = seed)
    if Gen.detected_event_count==0:
        continue
    Data = Gen.GetSurveyAndEventData()

    I = Inference(Data, gamma = True, vectorised = True, event_distribution_inf='Proportional', gauss=False, p_det=True,
                     survey_type='perfect', resolution_H_0=100, H_0_Min = 50, H_0_Max = 100, gamma_known = False, gauss_type = "Cartesian")
    # I.H_0_Prob()

    x, y = zip(*Gen.detected_coords)
    fig, main_ax = plt.subplots()
    main_ax.set_ylim(-Gen.size, Gen.size)
    main_ax.set_xlim(-Gen.size, Gen.size)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
    main_ax.add_patch(cutoff)
    for _ in range(Gen.n):
        main_ax.plot(Gen.distance_range[_, :, 0], Gen.distance_range[_, :, 1], "-", color="b", )
    for (x, y, s) in zip(x, y, Gen.detected_luminosities):
        main_ax.add_artist(plt.Circle(xy=(x, y), radius=s + 0.001 * Gen.L_star, color="k", zorder=3))
    main_ax.scatter(0, 0, s=Gen.size / 1.25, c="k", marker="x")
    main_ax.xaxis.set_tick_params(labelbottom=False)
    main_ax.yaxis.set_tick_params(labelleft=False)
    main_ax.set_xticks([])
    main_ax.set_yticks([])
    plt.tight_layout()

    if Gen.detected_event_count != 0:
        x, y = zip(*Gen.BH_true_coords[indices])
        for (xhat, yhat, s) in zip(*zip(*Gen.BH_detected_coords[indices]), Gen.BH_detected_luminosities[indices]):
            main_ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=Gen.L_star, color="red", zorder=4))
        if Gen.plot_contours is True:
            for i, Z in enumerate(Gen.BH_detected_meshgrid):
                if i in indices:
                    X, Y = Gen.BH_contour_meshgrid
                    z = Z
                    n = 1000
                    z = z / z.sum()
                    t = np.linspace(0, z.max(), n)
                    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
                    f = interpolate.interp1d(integral, t)
                    t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
                    main_ax.contour(X, Y, z, t_contours, colors="red", zorder=2)
        for (x, y, s) in zip(x, y, Gen.BH_true_luminosities[indices]):
            main_ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="g", zorder=4))

    xlim=[-92, 49]
    ylim=[-330, -140]

    y, height = 0.01, 0.45
    width = height*(xlim[-1]-xlim[0])/(ylim[-1]-ylim[0])
    x = 0.99 - width

    inset_ax = main_ax.inset_axes(
       [x, y, width, height],  # [x, y, width, height] w.r.t. axes
        xlim=xlim, ylim=ylim, # sets viewport & tells relation to main axes
        xticklabels=[], yticklabels=[]
    )

    coord = np.array([-33, -248])
    index = np.argmin(np.sum(np.square(Gen.detected_coords - coord),axis = 1))
    true_coord = Gen.detected_coords[index]

    x, y = zip(*Gen.detected_coords)
    cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
    inset_ax.add_patch(cutoff)
    for _ in range(Gen.n):
        inset_ax.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="g",)
    for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
        if i == index:
            neighbouring_galaxy = plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3)
        inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))
    inset_ax.scatter(0,0, s=Gen.size/1.25, c = "k", marker = "x")
    inset_ax.xaxis.set_tick_params(labelbottom=False)
    inset_ax.yaxis.set_tick_params(labelleft=False)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    plt.tight_layout()


    for i, (xhat, yhat, s) in enumerate(zip(*zip(*Gen.BH_detected_coords[indices]), Gen.BH_detected_luminosities[indices])):
        inset_ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=Gen.L_star, color="red", zorder = 4))
        if i == 3:
            detected_patch = plt.Circle(xy=(xhat, yhat), radius=Gen.L_star, color="red", zorder=4)



    for i, Z in enumerate(Gen.BH_detected_meshgrid):
        if i in indices:
            X, Y = Gen.BH_contour_meshgrid
            z = Z
            n = 1000
            z = z / z.sum()
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
            inset_ax.contour(X,Y, z, t_contours, colors="red", zorder = 2)

    x, y = zip(*Gen.BH_true_coords[indices])
    for i, (x, y, s) in enumerate(zip(x, y, Gen.BH_true_luminosities[indices])):
        inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="green", zorder = 4))
        if i == 3:
            true_patch = plt.Circle(xy=(x, y), radius=s, color="green", zorder=4)

    inset_ax.annotate(r'$\boldsymbol{Detected \; Location}$' "\n" r'$\boldsymbol{of \; GW \; Signal}$',xy=(Gen.BH_detected_coords[3]), xycoords='data',xytext=(-290,-270),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="red",
                        ec="none",
                        patchB=detected_patch,
                        connectionstyle="arc3,rad=0.1"), size = 16)

    inset_ax.annotate(r'$\boldsymbol{True \; Host \; Galaxy}$' "\n" r'$Unknown \; to \; Observer$',xy=(Gen.BH_true_coords[3]), xycoords='data',xytext=(-240,-310),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="green",
                        ec="none",
                        patchB=true_patch,
                        connectionstyle="arc3,rad=-0.15"), size = 16)


    inset_ax.annotate(r'$\boldsymbol{Potential \; Host}$' "\n" r'$\boldsymbol{Galaxy}$',xy=(true_coord), xycoords='data',xytext=(-110,-40),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="black",
                        patchB = neighbouring_galaxy,
                        ec="none",
                        connectionstyle="arc3,rad=0.1"), size = 16)

    inset_ax.annotate(r'$\boldsymbol{1, 2, 3 \,\sigma \, Contours}$' "\n" r'$\boldsymbol{of \; Host \; Location}$',xy=(20, -185), xycoords='data',xytext=(-100,0),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="red",
                        ec="none",
                        connectionstyle="arc3,rad=-0.5"), size = 16)

    inset_ax.annotate("\n",xy=(8, -197), xycoords='data',xytext=(-7.3,-12),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="red",
                        ec="none",
                        connectionstyle="arc3,rad=-0.35"), size = 16)

    inset_ax.annotate("\n",xy=(-2, -214), xycoords='data',xytext=(-7.3,-12),
                        arrowprops=dict(arrowstyle="wedge,tail_width=0.7",
                        fc="red",
                        ec="none",
                        connectionstyle="arc3,rad=-0.25"), size = 16)

    main_ax.indicate_inset_zoom(inset_ax, edgecolor="white")

    for axis in ['top','bottom','left','right']:
        main_ax.spines[axis].set_linewidth(3)

    # main_ax.indicate_inset_zoom(inset_ax, edgecolor="black", connector_lines = (True, True, True, True))

    # main_ax.set_title(str(seed))


plt.savefig("LowRes/2DAnnotatedUniverse.jpeg", dpi = 150)
plt.savefig("HighRes/2DAnnotatedUniverse.jpeg", dpi = 1500)

# careful! warn if aspect ratio of inset axes doesn't match main axes
# if not isclose(inset_ax._get_aspect_ratio(), main_ax._get_aspect_ratio()):
#     print("chosen inset x/ylim & width/height skew aspect w.r.t. main axes!")




# print(len(Gen.detected_redshifts))

# Universe_Plot = Sliding_Universe_3d(Gen)
# Universe_Plot.configure_traits()