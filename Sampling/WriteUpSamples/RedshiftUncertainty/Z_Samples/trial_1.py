'''
import numpy as np

n=100000
l = []
while len(l)<n:
    x = np.random.uniform(0,10,1)
    if x<0.1:
        l.append(x)

print(l[0])
print(len(l))
'''
'''
import multiprocessing
import time

def sleepy_man():
    print('Starting to sleep')
    time.sleep(1)
    print('Done sleeping')

if __name__ == '__main__':
    tic = time.time()
    p1 =  multiprocessing.Process(target= sleepy_man)
    p2 =  multiprocessing.Process(target= sleepy_man)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc-tic))

'''

import sys
sys.path.insert(0,'c:\\Users\\manco\\OneDrive\\Ambiente de Trabalho\\Masters_Project\\MsciProj')
from Sampling.ClassSamples import Sampler


import time
import multiprocessing 
print('import')


run = 2

#universe_number = 20
#universe_number = 40
universe_number = 50
#investigated_characteristic = 'redshift_multiprocessing'
#investigated_characteristic = 'corrected_redshift_multiprocessing'
#investigated_characteristic = 'better_corrected_redshift_multiprocessing'
#investigated_characteristic = 'Dan_hopefully_better_corrected_redshift_multiprocessing'
investigated_characteristic = 'Manuel_hopefully_better_corrected_redshift_multiprocessing'



#investigated_values = [0.005]
#investigated_values = [0.01]
max_numbers = []
#b = []
#f = []



'''
def sample_redshift(x):
    print('JOB JOB JOB {}'.format(x))
    name = investigated_characteristic + str(20*(run-1) + x)
    Investigation = Sampler(universe_count = 1, survey_type='imperfect', redshift_noise_sigma=investigated_values[0], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3, specify_gal_number=True, wanted_gal_n=500, wanted_det_events = 10, specify_event_number = True, 
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = name, investigated_value = investigated_values[0],
                                start_seed = 20*run + x, save_normally=2)
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)
'''

#wanted_gal_n = 2000
#wanted_det_events = 10
#d_ratio = 0.6
#beta = -1.3

investigated_values = [0.01]
wanted_gal_n = 3000
wanted_det_events = 10
d_ratio = 0.7
beta = -1.3


def sample_redshift(x):
    print('JOB JOB JOB {}'.format(50 + x))
    name = investigated_characteristic + str(50 + x)
    Investigation = Sampler(universe_count = 1, beta = beta, survey_type='imperfect', d_ratio = d_ratio, redshift_noise_sigma=investigated_values[0], resolution_H_0=100, H_0_Max=80, H_0_Min=60 , p_det=True, gamma = False, event_distribution='Proportional', total_luminosity=500/3,
                                specify_gal_number=True, wanted_gal_n=wanted_gal_n, wanted_det_events = wanted_det_events, specify_event_number = True, 
                                noise_distribution='BVMF_eff', event_distribution_inf='Proportional', investigated_characteristic = name, investigated_value = investigated_values[0],
                                start_seed = 50 + x, save_normally=2)
    Investigation.Sample()
    max_numbers.append(Investigation.max_num)


#def basic_func(x):
#    if x == 0:
#        return 'zero'
#    elif x%2 == 0:
#        return 'even'
#    else:
#        return 'odd'

#def multiprocessing_func(x):
#    y = x*x
#    time.sleep(2)
#    print('{} squared results in a/an {} number'.format(x, basic_func(y)))
    
if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(universe_number):
        p = multiprocessing.Process(target=sample_redshift, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
    print('That took {} seconds'.format(time.time() - starttime))