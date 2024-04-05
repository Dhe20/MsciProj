#%%

import numpy as np
import random
import scipy as sp
import scipy.special as sc
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.integrate import quad
from Visualising.Visualiser_3d import Visualiser_3d
from Tools import BB1_sampling as BB1Pack
import scipy.stats as ss
from Components.Universe import Universe
from Components.SurveyAndEventData import SurveyAndEventData
from Components.EventGenerator import EventGenerator


#%%

wanted_gal_n = 10000
beta = -1.3
event_rate = 1540.0
d_ratio = 0.4
BVM_c = 15
BVM_k = 2
wanted_det_events = 100


def gal_lum_factor():
    A = 1 + 1/0.1
    E = 1*(1+beta) * ((A**(2+beta) - 1)/(A**(2+beta) - A))
    print(E)
    return wanted_gal_n * E

total_luminosity = gal_lum_factor()

def factor():
    integral, err = quad(lambda x: 3 * (1 - 1/((1 + (d_ratio/x)**BVM_c)**BVM_k)) * (x**2) , 0, 1)
    req_time = wanted_det_events/(total_luminosity * event_rate * integral * np.pi/6)
    return req_time

sample_time = factor()

Gen = EventGenerator(dimension=3, cube=True, size=625, event_rate = event_rate, sample_time = sample_time, beta=beta, luminosity_gen_type="Full-Schechter", coord_gen_type="Random", cluster_coeff=0, characteristic_luminosity=1, lower_lim=0.1, total_luminosity=total_luminosity,
    BVM_c = 15, BVM_k = 2, BVM_kappa = 200, event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0, noise_std=0, plot_contours=False, seed=5)

d_all = np.linalg.norm(Gen.all_BH_true_coords, axis=1)
d_hat_all = np.linalg.norm(Gen.all_BH_detected_coords, axis=1)
d = np.linalg.norm(Gen.BH_true_coords, axis=1)
d_hat = np.linalg.norm(Gen.BH_detected_coords, axis=1)

#%%

import matplotlib
plt.style.use('default')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
ax.grid(ls='dashed', c='lightblue', alpha=0.8)

plt.scatter(d_all, d_hat_all, c='r', edgecolors='black', label='rejected')
plt.scatter(d,d_hat, c='limegreen', edgecolors='black', label='detected')
ax.hlines(y=d_ratio*625, xmin=5, xmax=1100, ls='dashed', color='black')

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 30, loc='upper left', framealpha=1)
ax.set_ylabel(r'$\hat{D}$ (Mpc)', fontsize=45, labelpad=15)
ax.set_xlabel(r'$D$ (Mpc)', fontsize=45, labelpad=15)
#ax.set_ylim(0.003,0.15)
ax.set_xlim(5,1050)
ax.set_ylim(5,1050)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)
plt.show()

# %%

#wanted_gal_n = 10000
beta = -1.5
event_rate = 1540.0
d_ratio = 0.4
BVM_c = 15
BVM_k = 2
wanted_det_events = 50

total_luminosity = 2000/3
sample_time = factor()

Gen = EventGenerator(dimension=3, cube=True, size=625, event_rate = event_rate, sample_time = sample_time, beta=beta, luminosity_gen_type="Full-Schechter", coord_gen_type="Random", cluster_coeff=0, characteristic_luminosity=1, lower_lim=0.1, total_luminosity=total_luminosity,
    BVM_c = 15, BVM_k = 2, BVM_kappa = 200, event_distribution="Proportional", noise_distribution="BVMF_eff", redshift_noise_sigma=0, noise_std=0, plot_contours=False, seed=5)



Data1 = Gen.GetSurveyAndEventData(min_flux=0.01/(4*np.pi*(0.4*625)**2))
Data2 = Gen.GetSurveyAndEventData(min_flux=0.1/(4*np.pi*(0.4*625)**2))
Data3 = Gen.GetSurveyAndEventData(min_flux=0.5/(4*np.pi*(0.4*625)**2))

f = [0.01, 0.1, 0.5]

def comp_f(distances, x, y):
    comp_f = []
    for i in x:
        comp_f.append(sum(distances<i)/sum(y<i))
    return comp_f

x = np.linspace(50, 1000, 200)
c1 = comp_f(np.linalg.norm(Data1.detected_coords, axis=1), x, np.linalg.norm(Gen.detected_coords, axis=1))
c2 = comp_f(np.linalg.norm(Data2.detected_coords, axis=1), x, np.linalg.norm(Gen.detected_coords, axis=1))
c3 = comp_f(np.linalg.norm(Data3.detected_coords, axis=1), x, np.linalg.norm(Gen.detected_coords, axis=1))

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

c = []
color = iter(cm.winter_r(np.linspace(0, 1, 3)))
for i in range(3):
    c.append(next(color))

ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)

ax.plot(x,c1, c=c[0], lw=4, label=r'$F_{{\mathrm{{min}}}}/F_* = {}$'.format(f[0]))
ax.plot(x,c2, c=c[1], lw=4, label=r'$F_{{\mathrm{{min}}}}/F_* = {}$'.format(f[1]))
ax.plot(x,c3, c=c[2], lw=4, label=r'$F_{{\mathrm{{min}}}}/F_* = {}$'.format(f[2]))
ax.legend(fontsize=30)
ax.grid(ls='dashed', c='lightblue', alpha=0.8, zorder=0)
ax.set_ylabel('Completeness fraction', fontsize=35, labelpad=15)
ax.set_xlabel(r'$D$ (Mpc)', fontsize=35, labelpad=15)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)

plt.show()

# %%
