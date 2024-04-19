#%%
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Components.Universe import Universe
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True


#%%

def corr_func(D, R, C, dR, S, m=1):
    # D and R are sets of point coordinates
    # Choose number of points used as centers C and delta_r
    bins = np.arange(m * dR, S, dR)
    DD = np.zeros(len(bins)-1)
    DR = np.zeros(len(bins)-1)
    for i in range(C):
        c = random.choice(D)
        #print(c)
        # calculate distances to that point
        dist_D = np.linalg.norm(D-c, axis=1)
        dist_R = np.linalg.norm(R-c, axis=1)
        DD += np.histogram(dist_D, bins=bins)[0]
        DR += np.histogram(dist_R, bins=bins)[0]
    #print(dist_D)
    y = (len(R)/len(D))*(DD/DR)-1

    #y = (DD*RR)/(np.linalg.norm(DR)**2) - 1
    return bins, y

def xi(r, gamma, r_0, e):
    return e*(r/r_0)**(-gamma)

# %%

#UD = Universe(dimension=2, total_luminosity=5000, characteristic_luminosity=1, lower_lim=0.1, coord_gen_type='Random', size=625, centroid_n=10, centroid_sigma=0.1)

#UD = Universe(dimension=2, total_luminosity=5000, characteristic_luminosity=1, lower_lim=0.1, coord_gen_type='Centroids', size=625, centroid_n=10, centroid_sigma=0.1)

UD = Universe(dimension=2, total_luminosity=5000, characteristic_luminosity=1, lower_lim=0.1, coord_gen_type='Clustered', cluster_coeff=1, size=625, centroid_n=10, centroid_sigma=0.1)

UR = Universe(dimension=2, total_luminosity=5000, characteristic_luminosity=1, lower_lim=0.1, coord_gen_type='Random', size=625)

D = UD.true_coords
R = UR.true_coords

#%%

b,y = corr_func(D,R,15000,0.25,UD.size, 10)

# %%

x = (b[:-1] + b[1:])/2

x_f = np.linspace(x[0], x[-1],10000)
popt, pcov = curve_fit(xi, x, y)
y_f = xi(x_f, *popt)

fig, ax = plt.subplots(figsize=(12,8))
ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax.scatter(x,y, s=3, c='magenta', label='Data')

ax.plot(x_f, y_f, ls='dashed', c='r', label='Power-law fit')
#ax.set_yscale('log')
#ax.set_xscale('log')

#plt.yscale('log')
#plt.xscale('log')
#plt.ylim(-0.5,3)
#plt.xlim(10,625)

ax.set_ylabel(r'$\xi(r)$', fontsize=40, labelpad=15)
ax.set_xlabel(r'$r$ (Mpc)', fontsize=40, labelpad=15)
#ax.legend(fontsize=27)
ax.set_xlim(-10,400)
ax.set_ylim(-0.5,6.5)

ax.set_ylim(-0.5,15.5)

plt.show()

print(popt)
print(np.sqrt(pcov))







# %%


for i in [40,80]:
    UD = Universe(dimension=2, total_luminosity=7000, coord_gen_type='Clustered', cluster_coeff=i, size=625, centroid_n=15, centroid_sigma=i)
    UR = Universe(dimension=2, total_luminosity=7000,coord_gen_type='Random', size=625)

    D = UD.true_coords
    R = UR.true_coords

    b,y = corr_func(D,R,5000,1,UD.size)

    x = (b[:-1] + b[1:])/2

    plt.scatter(x,y, s=3, c='turquoise', label=r'$\sigma_g/S=${}'.format(i))

    print(i)
    plt.legend()
    plt.show()

# %%
