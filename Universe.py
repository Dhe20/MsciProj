import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Galaxy import Galaxy
import powerbox as pbox

plt.style.use('dark_background')
class Universe:
    def __init__(self, galaxy_count = 1, dimension = 3, luminosity_gen_type = "Fixed", coord_gen_type = "Clustered", spacing = 1,
                 cluster_coeff = None):

        self.galaxy_count = galaxy_count
        self.dimension = dimension
        self.galaxies = np.empty((self.galaxy_count), dtype = object)
        self.luminosities = np.zeros((self.galaxy_count))
        self.coords = np.empty((self.galaxy_count,self.dimension))
        self.spacing = spacing
        self.cutoff = self.spacing*0.4
        self.cluster_coeff = cluster_coeff

        self.luminosity_generator = dict({"Random": self.random_luminosity, "Fixed": self.fixed_luminosity})
        self.coord_generator = dict({"Random": self.random_coords, "Clustered": self.clustered_coords})

        self.create_galaxies(luminosity_gen_type = luminosity_gen_type, coord_gen_type = coord_gen_type)

    def create_galaxies(self, luminosity_gen_type, coord_gen_type):
        self.coords = self.coord_generator[coord_gen_type](self.cluster_coeff)
        for i in range(self.galaxy_count):
            luminosity = self.luminosity_generator[luminosity_gen_type]()
            self.luminosities[i] = luminosity
            self.galaxies[i] = Galaxy(coords = self.coords[i], dimension = self.dimension, luminosity = luminosity)

    def fixed_luminosity(self):
        return 1

    def random_luminosity(self):
        return random.random()

    def random_coords(self, cluster_coeff):
        random_coords = np.zeros((self.galaxy_count, self.dimension))
        for i in range(self.galaxy_count):
            random_coords[i] = np.array([(random.random()-0.5)*self.spacing for _ in range(self.dimension)])
        return random_coords

    def clustered_coords(self, cluster_coeff):
        if cluster_coeff == None:
            cluster_coeff = 2

        power = lambda k: cluster_coeff * k ** -3


        lnpb = pbox.LogNormalPowerBox(
            N=512,  # Number of grid-points in the box
            dim=self.dimension,  # 2D box
            pk=power,  # The power-spectrum
            boxlength=1.,  # Size of the box (sets the units of k in pk)
            # seed=self.seed  # Set a seed to ensure the box looks the same every time (optional)
        )

        num_of_galaxies = self.galaxy_count
        clustered_sample = lnpb.create_discrete_sample(nbar=int(2 * num_of_galaxies),
                                                      randomise_in_cell=True,  # nbar specifies the number density
                                                      # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                                      )
        index_of_galaxies = list(np.arange(0, len(clustered_sample), 1))
        selected_index = random.sample(index_of_galaxies, k=num_of_galaxies)
        selected_galaxies = clustered_sample[selected_index, :]*self.spacing

        return selected_galaxies

    def plot_universe(self, show = True):
        x, y = zip(*self.coords)
        fig, ax = plt.subplots()

        ax.set_ylim(-0.5*self.spacing, 0.5*self.spacing)
        ax.set_xlim(-0.5*self.spacing, 0.5*self.spacing)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        cutoff = plt.Circle((0, 0), self.cutoff, color='w', ls="--", fill="")
        ax.add_patch(cutoff)
        for (x, y, s) in zip(x, y, self.luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="y"))
        ax.scatter(0,0, s=self.spacing/2.5, c = "w", marker = "x")
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
#
# Gen = Universe(spacing = 200, dimension = 2, galaxy_count=1000, luminosity_gen_type = "Fixed", coord_gen_type = "Clustered",
#                cluster_coeff=50)
# Gen.plot_universe()