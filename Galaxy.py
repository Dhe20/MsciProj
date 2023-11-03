import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Galaxy:
    def __init__(self, dimension = 3, luminosity = 1,
                 true_coords = np.array([0,0]), detected_coords = np.array([0,0])):
        self.dimension = dimension
        self.true_coords = true_coords
        self.detected_coords = detected_coords
        self.luminosity = luminosity
