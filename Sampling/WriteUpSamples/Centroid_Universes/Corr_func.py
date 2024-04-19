#%%
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Components.Universe import Universe

#%%

def corr_func(D, R, C, dR, S):
    # D and R are sets of point coordinates
    # Choose number of points used as centers C and delta_r
    bins = np.arange(dR, S, dR)
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

UD = Universe(dimension=2, total_luminosity=7000, coord_gen_type='Centroids', size=625, centroid_n=15, centroid_sigma=0.05)
UR = Universe(dimension=2, total_luminosity=7000,coord_gen_type='Random', size=625)

D = UD.true_coords
R = UR.true_coords

#%%

b,y = corr_func(D,R,5000,1,UD.size)

# %%

x = (b[:-1] + b[1:])/2

x_f = np.linspace(x[0], x[-1],10000)
popt, pcov = curve_fit(xi, x, y)
y_f = xi(x_f, *popt)


plt.scatter(x,y, s=2, c='turquoise', label='Data')
plt.plot(x_f, y_f, ls='dashed', c='r', label='Power-law fit')

#plt.yscale('log')
#plt.xscale('log')
#plt.ylim(-0.5,3)
#plt.xlim(10,625)

plt.ylabel(r'$\xi(r)$')
plt.xlabel(r'$r$ (Mpc)')
plt.legend()
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
