#%%

import numpy as np
import scipy as sp
import sympy as sym
from sympy import nsolve, beta, cos
import random
import pylab as pl
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import time as tt
from collections import Counter

#%%

rcParams['mathtext.fontset'] = 'dejavuserif'

rcParams["font.family"] = 'serif'
#rcParams['font.family'] = 'Times New Roman'
rcParams['xtick.major.pad'] = 5
rcParams['ytick.major.pad'] = 5
rcParams['axes.grid'] = False
rcParams["grid.linestyle"]= (5, (10, 3))
rcParams["grid.linewidth"]= 1
rcParams['grid.color'] = 'lightblue'
rcParams['font.size'] = 20
rcParams['figure.figsize'] = (12, 8)
rcParams['figure.titlesize'] = 50
rcParams['legend.fontsize'] = 23
rcParams['axes.titlesize'] = 35
rcParams['axes.labelsize'] = 40
rcParams['xtick.labelsize'] = 30
rcParams['ytick.labelsize'] = 30
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams["xtick.bottom"] = True 
rcParams["ytick.left"] = True
rcParams["xtick.top"] = True 
rcParams["ytick.right"] = True
rcParams['xtick.major.size'] = 7
rcParams['ytick.major.size'] = 7
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1
rcParams['xtick.minor.size'] = 5
rcParams['ytick.minor.size'] = 5
#rcParams['xtick.minor.width'] = 1
#rcParams['ytick.minor.width'] = 1
rcParams['axes.axisbelow'] = True

#%%

def d2_gauss(X, Y, u_x, u_y, s_x, s_y):
    Z = np.exp(-(((X-u_x)/s_x)**2 + ((Y-u_y)/s_y)**2)/2)/(2*np.pi*s_x*s_y)
    return Z

def von_misses(x, u, kappa):
    return np.exp(kappa*np.cos(x-u))/(2*np.pi*sp.special.iv(0,kappa))

def burr(x, c, k, l):
    return (c*k/l)*((x/l)**(c-1))*((1+(x/l)**c)**(-k-1))

class Universe:

    def __init__(self, dimension, size, total_luminosity, luminosity_func, max_distance, contours, noise_std, event_choice, alpha=1, characteristic_luminosity=1, min_lum=0, max_lum=1):
        self.dim = dimension
        self.L_0 = total_luminosity
        self.size = size
        self.luminosity_func = luminosity_func
        self.max_D = max_distance
        self.contours = contours
        self.noise_std = noise_std
        self.event_choice = event_choice
        self.alpha = alpha
        self.L_star = characteristic_luminosity
        self.min_L = min_lum
        self.max_L = max_lum

        self.x = np.linspace(-self.size, self.size, 50*self.size)
        self.y = np.linspace(-self.size, self.size, 50*self.size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        if self.luminosity_func == 'schechter':
            N_0 = self.L_0/(self.L_star*(1+self.alpha))
            self.n = round(N_0)
            rng = np.random.default_rng()
            self.gal_lum = rng.gamma(self.alpha, scale=self.L_star, size=self.n)
            if any(self.gal_lum==0.0):
                s = set(U.gal_lum) 
                self.gal_lum += sorted(s)[1]


        elif self.luminosity_func == 'cut_schechter':
            N_0 = self.L_0/(self.L_star*(1+self.alpha))
            self.n = round(N_0)
            rng = np.random.default_rng()
            gal_lum = []
            while len(gal_lum)<self.n:
                l = rng.gamma(self.alpha, scale=self.L_star, size=1)[0]
                if l > self.min_L:
                    gal_lum.append(l)
            self.gal_lum = np.array(gal_lum)

        elif self.luminosity_func == 'uniform':
            N_0 = self.L_0/(0.5*(self.min_L + self.max_L))
            self.n = round(N_0)
            self.gal_lum = np.random.uniform(0, 1, size=self.n)

        # No clustering yet
        self.galaxies = np.random.uniform(low = -size, high = size, size = (self.n, self.dim))

    def event(self):
        if self.event_choice == 'uniform':
            self.source = random.randint(0,self.n-1) 
        elif self.event_choice == 'proportional':
            n_list = list(np.arange(0,self.n))
            self.source = random.choices(n_list, weights=self.gal_lum)[0]
        
        self.source_loc = self.galaxies[self.source]
        rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=self.noise_std, size=self.dim)
        self.new_loc = self.galaxies[self.source] + noise
        D = np.linalg.norm(self.new_loc)
        while D > self.max_D:
            self.source = random.randint(0,self.n-1)
            self.source_loc = self.galaxies[self.source]
            noise = rng.normal(loc=0.0, scale=self.noise_std, size=self.dim)
            self.new_loc = self.galaxies[self.source] + noise
            D = np.linalg.norm(self.new_loc)
        self.noise = noise
        print(self.noise)
       

    def plot_event(self):

        self.event()
        print('Event chosen')

        fig, ax = plt.subplots(figsize=(12,8))

        Drawing_uncolored_circle = plt.Circle((0, 0), self.max_D, fill = False, linestyle='--') 
        ax.set_aspect(1)
        ax.add_artist(Drawing_uncolored_circle)
        
        Drawing_uncolored_circle_2 = plt.Circle((self.source_loc[0], self.source_loc[1]), np.mean(self.gal_lum), fill = False, linestyle='--', color='red') 
        ax.add_artist(Drawing_uncolored_circle_2)

        if self.contours == 'gaussian':
            u_x = self.new_loc[0]
            u_y = self.new_loc[1]
            s_x = self.noise_std
            s_y = self.noise_std
            Z = np.exp(-(((self.X-u_x)/s_x)**2 + ((self.Y-u_y)/s_y)**2)/2)/(2*np.pi*s_x*s_y)

            vals = [d2_gauss(u_x + 3*s_x, u_y + 3*s_y, u_x, u_y, s_x, s_y), d2_gauss(u_x + 2*s_x, u_y + 2*s_y, u_x, u_y, s_x, s_y), d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
            print(vals)
            #strs = ['3s', '2s', '1s']
            CS = ax.contour(self.X, self.Y, Z, vals, colors='red')
            print(CS.levels)
            #fmt = {}
            #for l, s in zip(CS.levels, strs):
            #    fmt[l] = s
        
        elif self.contours == 'shell':
            u_x = self.new_loc[0]
            u_y = self.new_loc[1]
            s_x = self.noise_std
            s_y = self.noise_std
            r = np.sqrt((self.X)**2 + (self.Y)**2)
            phi = np.arctan2(self.Y, self.X)
            c = 3
            k = 25
            kappa = 70
            u_r = np.sqrt((u_x)**2 + (u_y)**2)
            u_phi = np.arctan2(u_y, u_x)
            angular = von_misses(phi,u_phi,kappa)
            radial = burr(r,c,k,u_r)
            Z = r*angular*radial

            vals = [d2_gauss(u_x + 3*s_x, u_y + 3*s_y, u_x, u_y, s_x, s_y), d2_gauss(u_x + 2*s_x, u_y + 2*s_y, u_x, u_y, s_x, s_y), d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
            print(vals)
            #strs = ['3s', '2s', '1s']
            CS = ax.contour(self.X, self.Y, Z, vals, colors='red')
            print(CS.levels)

        #gal_sizes = 10*self.gal_lum/max(self.gal_lum) + 0.01
        gal_sizes = self.gal_lum + 0.001
        #ax.clabel(CS, CS.levels, fmt = fmt, inline=True, fontsize=10)
        ax.set_title('Universe')
        plt.scatter(self.galaxies[:,0], self.galaxies[:,1], s= gal_sizes, color='blue')
        #plt.scatter(self.source_loc[0], self.source_loc[1], s=20, color='red')
        print(u_x)
        plt.scatter(u_x, u_y, s=20, color='magenta', marker='x')
        plt.show()

#%%

#U = Universe(2, 50, 10000, 'schechter', 40, 'gaussian', 3, 'proportional', alpha=0.3, characteristic_luminosity=4)
#U = Universe(2, 1000, 50, 'schechter', 40, 1, 3, 'uniform')
U = Universe(2, 50, 10000, 'schechter', 40, 'shell', 3, 'proportional', alpha=0.3, characteristic_luminosity=4)

U.plot_event()


# %%