#%%

import sys
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler

from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
import scipy.stats as sps
from scipy.optimize import curve_fit
import seaborn as sn
from matplotlib import gridspec, collections
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

#%%

def bias_dist(x, H=70):
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps

def C_I_samp(x):
    H = np.random.uniform(50,100)
    H = 70
    ps = []
    inc = x.index[1]-x.index[0]
    cut_x = x.loc[x.index<=H]
    for i in cut_x.columns:
        ps.append(inc*np.sum(cut_x[i]))
    return ps

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def axis_namer(s):
    index = s.find('_')
    if index != -1:
        title = s[0].upper()+s[1:index]+' '+s[index+1].upper()+s[index+2:]
    else:
        title = s[0].upper()+s[1:]
    return title


#%%

# Q_0

investigated_characteristic = 'q_0'
investigated_values = [True]
max_numbers = []
#b = []
#f = []

Investigation = Sampler(universe_count = 10, beta=-1.3, hubble_law='quadratic', hubble_law_inf='quadratic', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=1000/3, wanted_det_events = 50, specify_event_number = True, 
                        noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[0])
Investigation.Sample()
max_numbers.append(Investigation.max_num)


# %%










#%%

size = 10*625
probs = 'Proportional'

R = 100
splits = 20
seed = 12

#%%


Gen = EventGenerator(dimension=3, cube=True, size=size, d_ratio = 0.5, event_rate = 1540.0, sample_time = 0.0005978, luminosity_gen_type="Full-Schechter", hubble_law='quadratic', coord_gen_type="Random", cluster_coeff=0, characteristic_luminosity=1, lower_lim=0.1, total_luminosity=1500,
BVM_c = 15, BVM_k = 2, BVM_kappa = 200, beta=-1.3, event_distribution=probs, noise_distribution="BVMF_eff", redshift_noise_sigma=0., noise_std=0, plot_contours=False, seed=seed)

Data = Gen.GetSurveyAndEventData()
I = Inference(Data, H_0_Min=55, H_0_Max=85, resolution_H_0=100, resolution_q_0=R, survey_type = 'perfect', hubble_law_inf = 'quadratic', gamma = False, event_distribution_inf = probs, flux_threshold=0)

self = I

print(len(Data.BH_detected_coords))

print(len(Data.detected_redshifts))

#%%


start_time = time.time()

for w in range(splits):   
    initial = w*int(R/splits)
    final = initial + int(R/splits)

    h_0_recip = np.reciprocal(self.H_0_range)
    H_0_recip = np.tile(h_0_recip, (int(R/splits),1)).T

    q_0 = np.tile(self.q_0_range[initial:final], (self.resolution_H_0,1))

    redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, int(R/splits), 1))

    Ds = redshifts * self.c * ( 1 + 0.5 * redshifts * (1-q_0[:,:,np.newaxis] )) * H_0_recip[:,:,np.newaxis]

    print("--- %s seconds ---" % (time.time() - start_time))

    burr_full = self.get_vectorised_burr_quadratic(Ds)

    print("--- %s seconds ---" % (time.time() - start_time))

    vmf = self.get_vectorised_vmf()

    print("--- %s seconds ---" % (time.time() - start_time))

    luminosity_term = self.lum_term[self.event_distribution_inf](redshifts, initial, final)

    full_expression = burr_full * vmf[:,:,np.newaxis,:] * luminosity_term

    print("--- %s seconds ---" % (time.time() - start_time))

    self.H_0_pdf_single_event = np.sum(full_expression, axis=3)
    #self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)
    '''
    if self.p_det:
        p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
        P_det_total = np.sum(p_det_vec, axis=1)
        self.P_det_total = P_det_total
        P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
        self.H_0_pdf = self.H_0_pdf/P_det_total_power
    '''
    '''
    if self.p_det:
        p_det_vec = self.get_p_det_vec(Ds) * luminosity_term
        P_det_total = np.sum(p_det_vec, axis=2)
        self.P_det_total = P_det_total
        self.H_0_pdf_single_event = self.H_0_pdf_single_event / P_det_total
    '''
    print("--- %s seconds ---" % (time.time() - start_time))

    '''
    f = np.vectorize(math.frexp)
    split = f(self.H_0_pdf_single_event)
    flo = split[0]
    ex = split[1]
    p_flo = np.prod(flo, axis=0)
    p_ex = np.sum(ex, axis=0)
    scaled_ex = p_ex - np.max(p_ex)
    scaled_flo = p_ex / p_flo.flatten()[np.argmax(p_ex)]
    H_0_pdf_temp = scaled_flo * (0.5 ** (-1 * scaled_ex))
    '''
    if self.p_det:
        p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
        P_det_total = np.sum(p_det_vec, axis=2)
        self.P_det_total = P_det_total
        #P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
        #self.H_0_pdf = self.H_0_pdf/P_det_total_power
        self.H_0_pdf_single_event = self.H_0_pdf_single_event/P_det_total

    self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
    
    if w==0:
        run_min = np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)])
    else:
        if np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)])< run_min:
            run_min = np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)])
    
    H_0_pdf_temp = np.exp(self.log_H_0_pdf - run_min)
    #H_0_pdf_temp /= np.sum(H_0_pdf_temp) * (self.H_0_increment)


    print("--- %s seconds ---" % (time.time() - start_time))

    if w == 0:
        self.H_0_pdf = H_0_pdf_temp
    else:
        self.H_0_pdf = np.concatenate((self.H_0_pdf, H_0_pdf_temp), axis=1)


self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment) * (self.q_0_increment)

#%%

self.H_0_samples.to_csv("SampleUniverse_"+ str(size)+ '_' + str(probs) + ".csv")


#%%


fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
im = ax.imshow(self.H_0_pdf , cmap = 'magma')
fig.colorbar(im, orientation='vertical')

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
h = np.sum(self.H_0_pdf, axis=1) * (self.q_0_increment)
h /= np.sum(h) * (self.H_0_increment)
ax.plot(self.H_0_range, h)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
q = np.sum(self.H_0_pdf, axis=0) * (self.H_0_increment)
q /= np.sum(q) * (self.q_0_increment)
ax.plot(self.q_0_range, q)

print(np.sum(self.H_0_range*h*self.H_0_increment))
print(np.sum(self.q_0_range*q*self.q_0_increment))

#%%

prior = sp.stats.norm.pdf(self.q_0_range, loc=-0.53, scale=0.5)

pdf = self.H_0_pdf*prior

#%%

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()

im = ax.imshow(self.H_0_pdf*prior , cmap = 'magma')
fig.colorbar(im, orientation='vertical')

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
h = np.sum(self.H_0_pdf*prior, axis=1)
ax.plot(self.H_0_range, h)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot()
q = np.sum(self.H_0_pdf*prior, axis=0)
ax.plot(self.q_0_range, q)


#%%


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.constrained_layout.use'] = True

fig = plt.figure(figsize = (12,16), layout='constrained')
# create grid for different subplots
spec = gridspec.GridSpec(ncols=2, nrows=3,
                        wspace=0.,
                         hspace=0.)


ax2 = fig.add_subplot(spec[1,0])
ax1 = fig.add_subplot(spec[0,0])#, sharex=ax2)
ax3 = fig.add_subplot(spec[1,1])#, sharey=ax2)
#ax4 = fig.add_subplot(spec[0,1])#, sharey=ax2)


im = ax2.contourf(self.H_0_range, self.q_0_range, pdf.T, 5, cmap='afmhot_r')
cbaxes = fig.add_axes([0.56, 0.7, 0.3, 0.05])
cbar = plt.colorbar(im, cax = cbaxes, orientation="horizontal")
cbar.ax.tick_params(labelsize=20) 
cbar.ax.set_title(r'$P\,(H_0,q_0\,|\,\mathrm{data})$', fontsize=30, pad=20)
ax2.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=30, labelpad=10)
ax2.set_ylabel(r'$q_0$',  fontsize=30, labelpad=10)

q = np.sum(self.H_0_pdf*prior, axis=0)
q /= np.sum(q)*self.q_0_increment
ax3.plot(q,self.q_0_range, c='b', lw=2)
ax3.plot(prior, self.q_0_range, c='tomato', lw=2, ls='dashed')
ax3.grid(ls='dashed', c='gray', alpha=0.8, axis='y')
ax3.set_yticklabels([])
ax3.set_xticklabels([])


h = np.sum(self.H_0_pdf*prior, axis=1)
h /= np.sum(h)*self.H_0_increment
ax1.plot(self.H_0_range, h, c='b', lw=2)
ax1.grid(ls='dashed', c='gray', alpha=0.8, axis='x')

ax1.set_xticklabels([])
ax1.set_yticklabels([])

ax2.tick_params(axis='both', which='major', direction='in', labelsize=18, size=4, width=1, pad = 12)
ax1.tick_params(axis='both', which='major', direction='in', labelsize=18, size=4, width=1, pad = 12)
ax3.tick_params(axis='both', which='major', direction='in', labelsize=18, size=4, width=1, pad = 12)


ax2.set_xlim(67,82)
ax1.set_xlim(67,82)

ax2.set_ylim(-1.8,-0.2)
ax3.set_ylim(-1.8,-0.2)

plt.show()

# %%
