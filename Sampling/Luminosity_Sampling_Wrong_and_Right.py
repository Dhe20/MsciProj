import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sps
from scipy.optimize import curve_fit
import seaborn as sn
from matplotlib import gridspec, collections
from Sampling.ClassSamples import Sampler
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'Calibri'
matplotlib.rcParams['figure.constrained_layout.use'] = True

f = 4*np.pi/375
c = 1.5*32*np.pi/3000

investigated_characteristic = 'trial_survey_completeness_correctly'
#investigated_values = [25,75,95]
investigated_values = np.array([0.2, 0.4, 0.6, 0.8]) #,0.5])
investigated_values /= (4*np.pi*(0.4*625)**2)
max_numbers = []
percentage = []


for i in tqdm(range(len(investigated_values))):
    for flux_thresh in [0, investigated_values[i]]:
        if flux_thresh == 0:
            investigated_characteristic = 'trial_survey_completeness_wrong'
        else:
            investigated_characteristic = 'trial_survey_completeness_correct'
        Investigation = Sampler(universe_count = 1000, min_flux=investigated_values[i], completeness_type='cut_lim', p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=2000/3, wanted_det_events = 50, specify_event_number = True,
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', lum_function_inf='Full-Schechter', flux_threshold = flux_thresh, investigated_characteristic = investigated_characteristic, investigated_value = investigated_values[i])
        Investigation.Sample()




