#%%

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns 
import scipy.stats as sps 
from matplotlib.ticker import MultipleLocator


#%%

plt.rcParams['text.usetex'] = 1

planck_u = 67.3
planck_sigma = 0.6
shoes_u = 74.2
shoes_sigma = 1.8

x = np.linspace(63,81,10000)
y_p = sps.norm.pdf(x, loc=planck_u, scale= planck_sigma)
y_s = sps.norm.pdf(x, loc=shoes_u, scale= shoes_sigma)

tension = 4.4
y_line = 0.35
d = 0.05

fig,ax = plt.subplots(1,1,figsize=(16,9))
ax.plot(x,y_p, c='r', label='Planck')
ax.plot(x,y_s, c='b', label='SHOES')

plt.hlines(y = y_line, xmin=planck_u, xmax=shoes_u, color = 'black', ls='dashed') 
plt.vlines(x = [planck_u,shoes_u], ymin=y_line-d, ymax=y_line+d, color = 'black', linestyle = '-') 
ax.annotate('${}\sigma$'.format(tension), xy=((planck_u + shoes_u)/2 , y_line+2*d), xytext=((planck_u + shoes_u)/2 , y_line+d), ha='center', fontsize=30)
ax.set_ylabel('P', fontsize=30)
ax.set_xlabel('$H_0$', fontsize=30)
ax.set_title('Early and Late Unverse $H_0$ measurements', fontsize=40, pad=20)
ax.grid(color='grey', ls='dashed')
ax.legend(loc='upper right',fontsize = 25, framealpha = 1)

ax.tick_params(axis='both',labelsize = 20, direction='in',top = False, right = False, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

plt.show()

# %%
