import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Galaxy:
    def __init__(self, dimension = 3, detected_luminosity = 1, true_luminosity = 1, flux = 1/1000,
                 true_coords = np.array([0,0]), detected_coords = np.array([0,0])):
        self.dimension = dimension
        self.true_coords = true_coords
        self.detected_coords = detected_coords
        self.detected_luminosity = detected_luminosity
        self.true_luminosity = true_luminosity
        self.flux = flux
