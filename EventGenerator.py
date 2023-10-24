from Universe import Universe
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy

class EventGenerator(Universe):
    def __init__(self, galaxy_count = 1, dimension = 3, luminosity_gen_type = "Fixed",
                 coord_gen_type = "Random", spacing = 1,
                 event_count = 1, event_distribution = "Random",
                 noise_distribution = "gauss"):
        super().__init__(galaxy_count, dimension, luminosity_gen_type,
                 coord_gen_type, spacing)
        self.event_distribution = event_distribution
        self.event_count = event_count
        self.resolution = 100
        self.confidence_levels = [0.9973,  0.9545, 0.6827]


        self.noise_distribution = noise_distribution
        self.noise_sigma = self.spacing/50
        self.BH_galaxies = np.empty((self.event_count), dtype = object)
        self.BH_coords = np.zeros((self.event_count, self.dimension))
        self.BH_luminosities = np.zeros((self.event_count))
        self.BH_detected_coords = np.empty((self.event_count, self.dimension))
        if self.dimension == 2:
            self.BH_contour_meshgrid = np.meshgrid(np.linspace(-self.spacing/2, self.spacing/2, self.resolution),
                                                   np.linspace(-self.spacing / 2, self.spacing / 2, self.resolution))
        if self.dimension == 3:
            self.BH_contour_meshgrid = np.meshgrid(np.linspace(-self.spacing/2, self.spacing/2, self.resolution),
                                                   np.linspace(-self.spacing / 2, self.spacing / 2, self.resolution),
                                                   np.linspace(-self.spacing / 2, self.spacing / 2, self.resolution))
        self.BH_detected_meshgrid = np.empty((self.event_count, *np.shape(self.BH_contour_meshgrid[0])))



        self.event_generator = dict({"Random": self.random_galaxy})
        self.coord_noise_generator = dict({"gauss": self.gauss_noise})
        self.generate_events(event_distribution = self.event_distribution,
                             noise_distribution = self.noise_distribution)

    def generate_events(self, event_distribution, noise_distribution):
        event_count = 0
        while event_count < self.event_count:
            selected = self.event_generator[event_distribution]()
            noise = self.coord_noise_generator[noise_distribution]()
            if np.sqrt(np.sum(np.square(self.coords[selected] + noise))) > self.cutoff:
                continue
            self.BH_galaxies[event_count] = self.galaxies[selected]
            self.BH_coords[event_count] = self.coords[selected]
            self.BH_detected_coords[event_count] = self.coords[selected] + noise
            self.BH_luminosities[event_count] = self.luminosities[selected]
            if self.dimension == 2:
                grid = self.gaussian_2d(*self.BH_contour_meshgrid,
                                                         self.BH_detected_coords[event_count], self.noise_sigma)
                self.BH_detected_meshgrid[event_count] = grid/np.max(grid)
            event_count+=1

    def random_galaxy(self):
        return random.randint(0, self.galaxy_count-1)

    def gauss_noise(self):
        covar = np.identity(self.dimension)*self.noise_sigma**2
        return np.random.multivariate_normal([0]*self.dimension, covar)

    def plot_universe_and_events(self, show = True):
        fig, ax = self.plot_universe(show = False)
        x, y = zip(*self.BH_coords)
        xhat, yhat = zip(*self.BH_detected_coords)
        for (xhat, yhat, s) in zip(xhat, yhat, self.BH_luminosities):
            ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=s, color="r"))
        for i, Z in enumerate(self.BH_detected_meshgrid):
            X, Y = self.BH_contour_meshgrid
            z = Z
            n = 1000
            z = z / z.sum()
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
            ax.contour(X,Y, z, t_contours, colors="r")
        for (x, y, s) in zip(x, y, self.BH_luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="g"))
        if show:
            plt.show()

    def gaussian_2d(self, x, y, mu, sigma):
        sig_x = sigma**2
        sig_y = sigma**2
        sig_xy = 0
        rv = scipy.stats.multivariate_normal(mu, [[sig_x, sig_xy], [sig_xy, sig_y]])
        Z = rv.pdf(np.dstack((x, y)))
        return Z



Gen = EventGenerator(event_count=5, spacing = 1000, dimension = 2, galaxy_count=1000,
                     luminosity_gen_type = "Random", coord_gen_type = "Random")

Gen.plot_universe_and_events()



