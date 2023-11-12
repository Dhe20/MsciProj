#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Galaxy import Galaxy
import powerbox as pbox
from BB1_sampling import *
from scipy.special import gamma, gammaincc


plt.style.use('dark_background')
class Universe:
    def __init__(self, dimension = 3, luminosity_gen_type = "Fixed", coord_gen_type = "Clustered",
                 cluster_coeff = 2, total_luminosity = 1000, size = 1,
                 alpha = .3, characteristic_luminosity = 1, min_lum = 0,
                 max_lum = .5, lower_lim=1, beta=-1.5):

        self.n = 0

        self.L_0 = total_luminosity
        self.dimension = dimension
        self.luminosity_func = luminosity_gen_type

        self.alpha = alpha
        self.size = size
        self.L_star = characteristic_luminosity
        self.min_L = min_lum
        self.max_L = max_lum
        self.lower_lim = lower_lim
        self.beta = beta

        self.cluster_coeff = cluster_coeff
        self.max_D = self.size*0.7

        self.luminosity_generator = dict({"Uniform": self.uniform_galaxies,
                                          "Fixed": self.fixed_luminosity,
                                          "Cut-Schechter": self.cut_schechter,
                                          "Shoddy-Schechter":self.schecter_luminosity,
                                          "Full-Schechter":self.full_schechter
                                          })

        self.luminosities = self.luminosity_generator[luminosity_gen_type]()

        # self.galaxies = np.empty((self.n), dtype = object)

        self.coord_generator = dict({"Random": self.random_coords,
                                     "Clustered": self.clustered_coords})

        self.coords = self.coord_generator[coord_gen_type]()

        # self.galaxies[i] = Galaxy(coords=self.coords[i], dimension=self.dimension, luminosity=luminosity)


        # self.x = np.linspace(-self.size, self.size, 50*self.size)
        # self.y = np.linspace(-self.size, self.size, 50*self.size)
        # self.X, self.Y = np.meshgrid(self.x, self.y)






        # self.create_galaxies(luminosity_gen_type = luminosity_gen_type, coord_gen_type = coord_gen_type)


    def upper_inc_gamma(self, a, x):
        return gamma(a)*gammaincc(a,x)

    def fixed_luminosity(self):
        self.n = round(self.L_0/self.L_star)
        return [self.L_star]*self.n

    def schecter_luminosity(self):
        N_0 = self.L_0 / (self.L_star * (self.alpha))
        self.n = round(N_0)
        rng = np.random.default_rng()
        self.gal_lum = rng.gamma(self.alpha, scale=self.L_star, size=self.n)
        # if any(self.gal_lum == 0.0):
        #     s = set(self.gal_lum)
        #     self.gal_lum += sorted(s)[1]
        return self.gal_lum

    def cut_schechter(self):
        r = self.min_L/self.L_star
        E_L = self.L_star*self.alpha + self.L_star*(r**self.alpha)*(np.exp(-r))/self.upper_inc_gamma(self.alpha, r)
        N_0 = self.L_0 / E_L

        self.n = round(N_0)
        rng = np.random.default_rng()
        gal_lum = []
        while len(gal_lum) < self.n:
            l = rng.gamma(self.alpha, scale=self.L_star, size=1)[0]
            if l > self.min_L:
                gal_lum.append(l)
        self.gal_lum = np.array(gal_lum)
        return self.gal_lum

    def full_schechter(self):
        A = 1 + self.L_star/self.lower_lim
        if self.beta>-2 and self.beta!=-1:
            E_L = self.L_star*(1 + self.beta)*((A**(2+self.beta) - 1)/(A**(2+self.beta) - A))
        elif self.beta == -1:
            E_L = (1/(np.log(A)))*(self.L_star**2/(self.L_star + self.lower_lim))
        N_0 = self.L_0 / E_L
        # Should try the algo cited in paper
        self.n = round(N_0)
        BB1 = Full_Schechter(name="BB1", a=0.0)
        samples = BB1.rvs(b=2+self.beta,u=self.L_star,l=self.lower_lim, size = self.n)
        self.gal_lum = samples
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

    def plot_universe(self, show = True):
        x, y = zip(*self.coords)
        fig, ax = plt.subplots()

        ax.set_ylim(-self.size, self.size)
        ax.set_xlim(-self.size, self.size)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        cutoff = plt.Circle((0, 0), self.max_D, color='w', ls="--", fill="")
        ax.add_patch(cutoff)
        for (x, y, s) in zip(x, y, self.luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s+0.001*self.L_star, color="y"))
        ax.scatter(0,0, s=self.size/1.25, c = "w", marker = "x")
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.yaxis.set_tick_params(labelleft=False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

#%%
'''
Gen = Universe(size = 50, dimension = 2,
               luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
               cluster_coeff=50, characteristic_luminosity=0.1, lower_lim=0.05, min_lum=0.05, total_luminosity=500)
Gen.plot_universe()
'''
# dict({"Uniform": self.uniform_galaxies,
#                   "Fixed": self.fixed_luminosity,
#                   "Cut-Schecter": self.cut_schechter,
#                   "Shoddy-Schecter":self.schecter_luminosity,
#                   })
# %%
