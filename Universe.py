import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Galaxy import Galaxy
plt.style.use('dark_background')
class Universe:
    def __init__(self, galaxy_count = 1, dimension = 3, luminosity_gen_type = "Fixed", coord_gen_type = "Random", spacing = 1):
        self.galaxy_count = galaxy_count
        self.dimension = dimension
        self.galaxies = np.empty((self.galaxy_count), dtype = object)
        self.luminosities = np.zeros((self.galaxy_count))
        self.coords = np.empty((self.galaxy_count,self.dimension))
        self.spacing = spacing
        self.cutoff = self.spacing*0.4

        self.luminosity_generator = dict({"Random": self.random_luminosity, "Fixed": self.fixed_luminosity})
        self.coord_generator = dict({"Random": self.random_coords})

        self.create_galaxies(luminosity_gen_type = luminosity_gen_type, coord_gen_type = coord_gen_type)

    def create_galaxies(self, luminosity_gen_type, coord_gen_type):
        for i in range(self.galaxy_count):
            luminosity = self.luminosity_generator[luminosity_gen_type]()
            coords = self.coord_generator[coord_gen_type]()
            self.coords[i] = coords
            self.luminosities[i] = luminosity
            self.galaxies[i] = Galaxy(coords = coords, dimension = self.dimension, luminosity = luminosity)

    def fixed_luminosity(self):
        return 1

    def random_luminosity(self):
        return random.random()

    def random_coords(self):
        return np.array([(random.random()-0.5)*self.spacing for _ in range(self.dimension)])

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

Gen = Universe(spacing = 200, dimension = 2, galaxy_count=1000, luminosity_gen_type = "Fixed", coord_gen_type = "Random")
Gen.plot_universe()