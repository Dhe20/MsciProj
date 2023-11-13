from Universe import Universe
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy as sp
from SurveyAndEventData import SurveyAndEventData


class EventGenerator(Universe):
    def __init__(self, dimension = 2, luminosity_gen_type = "Fixed",
                 coord_gen_type = "Clustered",
                 cluster_coeff = 2, total_luminosity = 1000, size = 1,
                 alpha = .3, characteristic_luminosity = 1, min_lum = 0,
                 max_lum = 1, event_count = 1, event_distribution = "Random",
                 noise_distribution = "Gauss", contour_type = "BVM",
                 noise_std = 3, resolution = 400, BVM_c = 20,
                BVM_k = 1.5, BVM_kappa = 400, redshift_noise_sigma = 0):

        super().__init__(dimension = dimension, luminosity_gen_type = luminosity_gen_type,
                         coord_gen_type = coord_gen_type,
                         cluster_coeff = cluster_coeff, total_luminosity = total_luminosity,
                         size = size, alpha = alpha, characteristic_luminosity = characteristic_luminosity,
                         min_lum = min_lum, max_lum = max_lum, redshift_noise_sigma = redshift_noise_sigma
                         )


        self.event_distribution = event_distribution
        self.event_count = event_count
        self.resolution = resolution
        self.confidence_levels = [0.9973,  0.9545, 0.6827]

        self.event_choice = event_distribution

        self.noise_distribution = noise_distribution
        self.noise_sigma = noise_std
        self.BH_galaxies = np.empty((self.event_count), dtype = object)
        self.BH_true_coords = np.zeros((self.event_count, self.dimension))
        self.BH_detected_luminosities = np.zeros((self.event_count))
        self.BH_true_luminosities = np.zeros((self.event_count))
        self.BH_detected_coords = np.empty((self.event_count, self.dimension))

        self.contour_type = contour_type

        self.BVM_k = BVM_k
        self.BVM_c = BVM_c
        self.BVM_kappa = BVM_kappa



        if self.dimension == 2:
            self.BH_contour_meshgrid = np.meshgrid(np.linspace(-self.size, self.size, self.resolution),
                                                   np.linspace(-self.size, self.size, self.resolution))
        if self.dimension == 3:
            self.BH_contour_meshgrid = np.meshgrid(np.linspace(-self.size, self.size, self.resolution),
                                                   np.linspace(-self.size, self.size, self.resolution),
                                                   np.linspace(-self.size, self.size, self.resolution))
        self.BH_detected_meshgrid = np.empty((self.event_count, *np.shape(self.BH_contour_meshgrid[0])))



        self.event_generator = dict({"Random": self.random_galaxy,
                                     "Proportional":self.proportional_galaxy})
        self.coord_noise_generator = dict({"Gauss": self.gauss_noise})
        self.contour_generator = dict({"Gauss": self.gaussian_2d,
                                       "BVM": self.BVMShell})
        self.generate_events(event_distribution = self.event_distribution,
                             noise_distribution = self.noise_distribution)

    def generate_events(self, event_distribution, noise_distribution):
        event_count = 0
        while event_count < self.event_count:
            selected = self.event_generator[event_distribution]()
            noise = self.coord_noise_generator[noise_distribution]()
            if np.sqrt(np.sum(np.square(self.true_coords[selected] + noise))) > self.max_D:
                continue
            self.BH_galaxies[event_count] = self.galaxies[selected]
            self.BH_true_coords[event_count] = self.true_coords[selected]
            self.BH_detected_coords[event_count] = self.true_coords[selected] + noise
            self.BH_true_luminosities[event_count] = self.true_luminosities[selected]
            self.BH_detected_luminosities[event_count] = self.detected_luminosities[selected]
            if self.dimension == 2:
                grid = self.contour_generator[self.contour_type](*self.BH_contour_meshgrid,
                                                        self.BH_detected_coords[event_count],
                                                        self.noise_sigma)
                self.BH_detected_meshgrid[event_count] = grid/grid.sum()
            event_count+=1

    def random_galaxy(self):
        return random.randint(0, self.n-1)

    def proportional_galaxy(self):
        n_list = list(np.arange(0,self.n))
        source = random.choices(n_list, weights=self.true_luminosities)[0]
        return source


    def gauss_noise(self):
        rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=self.noise_sigma, size=self.dimension)
        return noise

    def plot_universe_and_events(self, show = True):
        fig, ax = self.plot_universe(show = False)
        x, y = zip(*self.BH_true_coords)
        xhat, yhat = zip(*self.BH_detected_coords)
        for (xhat, yhat, s) in zip(xhat, yhat, self.BH_true_luminosities):
            ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=s, color="r", zorder = 4))
        for i, Z in enumerate(self.BH_detected_meshgrid):
            X, Y = self.BH_contour_meshgrid
            z = Z
            n = 1000
            z = z / z.sum()
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
            ax.contour(X,Y, z, t_contours, colors="r", zorder = 2)
        for (x, y, s) in zip(x, y, self.BH_true_luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="g", zorder = 4))
        if show:
            plt.show()

    def gaussian_2d(self, x, y, mu, sigma):
        sig_x = self.noise_sigma**2
        sig_y = self.noise_sigma**2
        sig_xy = 0
        rv = sp.stats.multivariate_normal(mu, [[sig_x, sig_xy], [sig_xy, sig_y]])
        Z = rv.pdf(np.dstack((x, y)))
        return Z

    def d2_gauss(self, X, Y, u_x, u_y, s_x, s_y):
        Z = np.exp(-(((X-u_x)/s_x)**2 + ((Y-u_y)/s_y)**2)/2)/(2*np.pi*s_x*s_y)
        return Z

    def von_misses(self, x, u, kappa):
        return np.exp(kappa*np.cos(x-u))/(2*np.pi*sp.special.iv(0,kappa))

    def burr(self, x, c, k, l):
        return (c*k/l)*((x/l)**(c-1))*((1+(x/l)**c)**(-k-1))

    def BVMShell(self, x, y, mu, sigma):
        u_x = mu[0]
        u_y = mu[1]
        s_x = self.noise_sigma
        s_y = self.noise_sigma
        X = self.BH_contour_meshgrid[0]
        Y = self.BH_contour_meshgrid[1]
        r = np.sqrt((X) ** 2 + (Y) ** 2)
        phi = np.arctan2(Y, X)
        u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2)
        u_phi = np.arctan2(u_y, u_x)

        k = self.BVM_k
        c = self.BVM_c
        kappa = self.BVM_kappa

        angular = self.von_misses(phi, u_phi, kappa)
        radial = self.burr(r, c, k, u_r)
        Z = r * angular * radial

        vals = [self.d2_gauss(u_x + 3 * s_x, u_y + 3 * s_y, u_x, u_y, s_x, s_y),
                self.d2_gauss(u_x + 2 * s_x, u_y + 2 * s_y, u_x, u_y, s_x, s_y),
                self.d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
        return Z

    def GetSurveyAndEventData(self):
        return SurveyAndEventData(dimension = self.dimension, detected_coords = self.detected_coords,
                                  detected_luminosities = self.detected_luminosities,
                                  fluxes = self.fluxes, BH_detected_coords = self.BH_detected_coords, BVM_k = self.BVM_k,
                                  BVM_c = self.BVM_c, BVM_kappa = self.BVM_kappa, BurrFunc = self.burr, VonMissesFunc= self.von_misses,
                                  detected_redshifts=self.detected_redshifts)



# Gen = EventGenerator(dimension = 2, size = 50, event_count=2,
#                      luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Clustered",
#                      cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=400,
#                      event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = .0)
#
#
# Gen.plot_universe_and_events()



