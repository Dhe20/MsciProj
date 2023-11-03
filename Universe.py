import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Galaxy import Galaxy
import powerbox as pbox

plt.style.use('dark_background')
class Universe:
    def __init__(self, dimension = 3, luminosity_gen_type = "Fixed", coord_gen_type = "Clustered",
                 cluster_coeff = 2, total_luminosity = 1000, size = 1,
                 alpha = .3, characteristic_luminosity = 1, min_lum = 0,
                 max_lum = 1, H_0 = 70, redshift_noise_sigma=0.25):

        self.H_0 = H_0

        self.redshift_noise_sigma = redshift_noise_sigma

        self.L_0 = total_luminosity
        self.dimension = dimension
        self.luminosity_func = luminosity_gen_type

        self.alpha = alpha
        self.size = size
        self.L_star = characteristic_luminosity
        self.min_L = min_lum
        self.max_L = max_lum

        self.cluster_coeff = cluster_coeff
        self.max_D = self.size*0.7
        # self.n = self.L_0 / self.L_star

        self.luminosity_generator = dict({"Uniform": self.uniform_galaxies,
                                          "Fixed": self.fixed_luminosity,
                                          "Cut-Schecter": self.cut_schechter,
                                          "Shoddy-Schecter":self.schecter_luminosity,
                                          })

        self.luminosities = self.luminosity_generator[luminosity_gen_type]()

        self.galaxies = np.empty((self.n), dtype = object)

        self.coord_generator = dict({"Random": self.random_coords,
                                     "Clustered": self.clustered_coords})

        self.true_coords = self.coord_generator[coord_gen_type]()

        for i in range(self.n):
            self.galaxies[i] = Galaxy(true_coords=self.true_coords[i],
                                      dimension=self.dimension,
                                      luminosity=self.luminosities[i])

        rng = np.random.default_rng()
        self.detected_coords, self.distance_range = self.distance_error(rng)



    def fixed_luminosity(self):
        self.n = round(self.L_0/self.L_star)
        return [self.L_star]*self.n

    def schecter_luminosity(self):
        N_0 = self.L_0 / (self.L_star * (1 + self.alpha))
        self.n = round(N_0)
        rng = np.random.default_rng()
        self.gal_lum = rng.gamma(self.alpha, scale=self.L_star, size=self.n)
        # if any(self.gal_lum == 0.0):
        #     s = set(self.gal_lum)
        #     self.gal_lum += sorted(s)[1]
        return self.gal_lum

    def cut_schechter(self):
        N_0 = self.L_0 / (self.L_star * (1 + self.alpha))
        self.n = round(N_0)
        rng = np.random.default_rng()
        gal_lum = []
        while len(gal_lum) < self.n:
            l = rng.gamma(self.alpha, scale=self.L_star, size=1)[0]
            if l > self.min_L:
                gal_lum.append(l)
        self.gal_lum = np.array(gal_lum)
        return self.gal_lum

    def uniform_galaxies(self):
        N_0 = self.L_0 / (0.5 * (self.min_L + self.max_L))
        self.n = round(N_0)
        self.gal_lum = np.random.uniform(0, 1, size=self.n)
        return self.gal_lum

    def random_coords(self):
        random_coords = np.zeros((self.n, self.dimension))
        for i in range(self.n):
            random_coords[i] = np.array([(random.random()-0.5)*2*self.size for _ in range(self.dimension)])
        return random_coords

    def clustered_coords(self):

        power = lambda k: self.cluster_coeff * k ** -3

        lnpb = pbox.LogNormalPowerBox(
            N=512,  # Number of grid-points in the box
            dim=self.dimension,  # 2D box
            pk=power,  # The power-spectrum
            boxlength=1.,  # Size of the box (sets the units of k in pk)
            # seed=self.seed  # Set a seed to ensure the box looks the same every time (optional)
        )

        num_of_galaxies = self.n
        clustered_sample = lnpb.create_discrete_sample(nbar=int(2 * num_of_galaxies),
                                                      randomise_in_cell=True,  # nbar specifies the number density
                                                      # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                                      )
        index_of_galaxies = list(np.arange(0, len(clustered_sample), 1))
        selected_index = random.sample(index_of_galaxies, k=num_of_galaxies)
        selected_galaxies = clustered_sample[selected_index, :]*self.size*2

        return selected_galaxies

    def distance_error(self, rng):
        detected_coords = np.zeros((self.n, self.dimension))
        distance_range = np.zeros((self.n, 2, self.dimension))
        for i in range(self.n):
            r = np.sqrt(np.sum(np.square(self.galaxies[i].true_coords)))
            rhat = self.galaxies[i].true_coords / r
            sigma = self.redshift_noise_sigma*(r/(np.sqrt(self.dimension)*self.size))
            noise = rng.normal(loc=0.0, scale=sigma, size=1)
            detected_coords[i][:] = self.galaxies[i].true_coords + rhat*noise
            distance_range[i][:] = np.array([self.galaxies[i].true_coords - rhat*sigma*3,
                                             self.galaxies[i].true_coords + rhat*sigma*3])
            self.galaxies[i].detected_coords = detected_coords[i]
        return detected_coords, distance_range

    def plot_universe(self, show = True):
        x, y = zip(*self.detected_coords)
        fig, ax = plt.subplots()
        ax.set_ylim(-self.size, self.size)
        ax.set_xlim(-self.size, self.size)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        cutoff = plt.Circle((0, 0), self.max_D, color='w', ls="--", fill="")
        ax.add_patch(cutoff)
        for _ in range(self.n):
            ax.plot(self.distance_range[_,:,0], self.distance_range[_,:,1], "-", color="b",)
        for (x, y, s) in zip(x, y, self.luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*self.L_star, color="white", zorder = 3))
        ax.scatter(0,0, s=self.size/1.25, c = "w", marker = "x")
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
#
Gen = Universe(size = 50, dimension = 2,
               luminosity_gen_type = "Cut-Schecter", coord_gen_type = "Clustered",
               cluster_coeff=0, characteristic_luminosity=.1, total_luminosity=1000)
Gen.plot_universe()

