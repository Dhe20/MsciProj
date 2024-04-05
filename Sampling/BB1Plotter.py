#%%
import matplotlib.pyplot as plt
import powerbox as pbox
import numpy as np
import random
import time
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from Tools.BB1_sampling import *
def bb1_lower(x,b,u,l):
    b = b-2
    if b>-2 and b!=-1:
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
    elif b==-1:
        C = 1/(np.log(1+u/l))
    return (C/(l*(u**(1+b))))*x**(1+b)

def bb1_mid(x,b,u,l):
    b = b-2
    if b>-2 and b!=-1:
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
    elif b==-1:
        C = 1/(np.log(1+u/l))
    return (C/u)*((x/u)**b)#*np.exp(-x/u)


# %%

N = 100000
x=np.linspace(1/N,10,N)
y = p(x,0.7,1,0.001)

s = random.Random(1)
samp = Complete_BB1_rej(s,-1.3,1,0.001,100000)

#%%

s2 = np.random.default_rng(seed = 1)
samp2 = s2.gamma(0.7, scale=1, size=100000)
y2 = ss.gamma.pdf(x,0.7)


beta = -1.3
x_l = np.linspace(1/N, 10*0.001, 10000)
x_m = np.linspace(0.001*0.1, 10*1, 10000)
y_l = bb1_lower(x_l,0.7,1,0.001)
y_m = bb1_mid(x_m,0.7,1,0.001)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
ax.grid(ls='dashed', c='k', alpha=0.4)

ax.plot(x,y,ls='dashed',lw=4, c='k', label='Theoretical BB1', zorder = 3)
ax.plot(x_l,y_l,ls='dashed',lw=4, c='purple', label=r'Faint end $\sim L^{\beta+1}$', zorder = 3)
ax.plot(x_m,y_m,ls='dashed',lw=4, c='dodgerblue', label=r'Middle range $\sim L^{\beta}$', zorder = 3)

#ax.plot(x,y2,ls='dashed',lw=4, c='b', label='Theoretical Schechter')

ax.hist(samp, histtype='step', edgecolor='#e16462', density = True, bins=np.logspace(-5,3,25), lw=3, label='BB1 samples', hatch='//', zorder = 2)
#ax.hist(samp2, density = True, bins=np.logspace(-5,3,30), color='b', alpha=0.5, label='Schechter samples')


ax.tick_params(axis='both', which='major', direction='in', labelsize=30, size=8, width=3, pad = 9)
ax.legend(fontsize = 23, loc='lower left', framealpha=1)
ax.set_ylabel(r'$\Phi(L)/n_*$', fontsize=45, labelpad=15)
ax.set_xlabel(r'$L/L_*$', fontsize=45, labelpad=15)
#ax.set_ylim(0.003,0.15)
ax.set_xlim(1/10**5,20)
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_title('Individual and combined posteriors', fontsize=40, pad=30)

image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Plots//BB1Plotter.svg'

plt.savefig(image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)

plt.show()