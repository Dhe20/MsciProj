from Components.EventGenerator import EventGenerator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.integrate import quad
from scipy.stats import poisson
plt.style.use('default')
from scipy import interpolate
from tqdm import tqdm

Gen = EventGenerator(dimension = 2, size = 625, sample_time=0.01*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=400, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()



for H_0_num, H0 in tqdm(enumerate(np.linspace(50,100, 100))):
    fig = plt.figure(figsize=(9, 9))

    ax0 = plt.subplot()
    ax1=None
    ax2=None

    axs = [ax0, ax1, ax2]


    ax0.add_patch(plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill=""))
    x, y = zip(*Gen.detected_coords)
    cutoff = plt.Circle((0, 0), Gen.max_D, color='k', ls="--", fill="")
    ax0.add_patch(cutoff)
    for _ in range(Gen.n):
        ax0.plot(Gen.distance_range[_,:,0], Gen.distance_range[_,:,1], "-", color="k",)


    for i in range(Gen.detected_event_count):
        mu = Gen.BH_detected_coords[i]
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
        ax0.contour(X,Y, z, t_contours, colors="red", zorder = 2)
        ax0.scatter(Gen.BH_detected_coords[i, 0], Gen.BH_detected_coords[i, 1], s=200, color='r', marker='x', zorder=5)

    ax0.set_xticks([])  # Remove x-ticks
    ax0.set_yticks([])

    for i, (x, y, s) in enumerate(zip(x, y, Gen.detected_luminosities)):
        rad = Gen.c * Gen.detected_redshifts[i]/H0
        phi = np.arctan2(x, y)
        x = rad*np.sin(phi)
        y = rad*np.cos(phi)
        s = Gen.fluxes[i]*2*np.pi*rad
        ax0.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*Gen.L_star, color="k", zorder = 3))

    image_format = 'png'
    image_name = '/Users/daneverett/PycharmProjects/MSciProject/Visualising/Simulation_Explanation_2/Observer_Information_Gif/Observer_Information_{}.png'.format(H_0_num)
    plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=600)
    plt.close(fig)
