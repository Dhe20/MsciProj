#%%
import numpy as np
import random
import scipy.stats as ss
from scipy.optimize import fmin, fsolve
from math import gamma, modf



def neg_gamma(z):
    # Full analytic continuation of gamma function
    # to negative axis - uses Recurrence formula
    n = - modf(z)[1]+1
    p_l = np.arange(z,z+n+1)
    g = gamma(z+n+1)/np.prod(p_l)
    return g

def p(x,b,u,l):
    b = b-2
    if b>-2 and b!=-1:
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
    elif b==-1:
        C = 1/(np.log(1+u/l))
    p =(C/u)*(1-np.exp(-x/l))*((x/u)**b)*np.exp(-x/u)
    return p





# Define the distribution using rv_continuous
class Full_Schechter(ss.rv_continuous): 
    def _pdf(self, x, b,u,l):
        return p(x,b,u,l)#(1.0/const) * p(x)


'''

BB1 = Full_Schechter(name="BB1", a=0.0)

b = 0.5
u = 5
l = 2

# create pdf, cdf, random samples
#pdf = BB1.pdf(x = x, b=b,u=u,l=l,c=c)
#cdf = BB1.cdf(x = x, b=b,u=u,l=l,c=c)
samples = BB1.rvs(b=b,u=u,l=l, size = 1000)
'''


'''#%%
import matplotlib.pyplot as plt

x = np.linspace(0.00000001, 10.0, 100000000)

w = p(x,b,u,l)
w = w/np.sum(w)
a = np.random.choice(x, 1000, p=w)
plt.hist(samples, 100)

'''

def sample_schecther(pdf, low_l, up_l, samp_num, args, argmax_guess = 1):
    #min_x = fmin(lambda x: p(x,*args), argmin_guess)[0]
    max_x = fmin(lambda x: -pdf(x,*args), argmax_guess)[0]
    #min_y = pdf(min_x, *args)
    max_y = pdf(max_x, *args)
    samples = []
    while len(samples) < samp_num:
        x = np.random.uniform(low_l, up_l)
        accept_probs = pdf(x,*args)
        u = np.random.uniform(0,max_y)
        if u < accept_probs:
            samples.append(x)
    return samples

def test_pdf(x,c):
    if x<=2 and x>=0:
        return c*x
    else:
        return 0


# Specific to BB1

def max_eq_BB1(x,b,u,l):
    return (l*x-b*l*u)*np.exp(x/l) - (u+l)*x + b*l*u

def max_eq_quad(x,c):
    return -2*x+4


def rej_sampling(pdf,args,low_l,up_l,max_y,samp_num):
    samples = []
    while len(samples) < samp_num:
        x = np.random.uniform(low_l, up_l)
        accept_probs = pdf(x,*args)
        u = np.random.uniform(0,max_y)
        if u < accept_probs:
            samples.append(x)
    return samples

def sample_full_schecther(pdf, max_eq, args, low_l, up_l, samp_num):
    max_x = fsolve(max_eq, np.array([1]), args)[0]
    max_y = pdf(max_x, *args)
    if max_y == 0:
        max_y = pdf(1/(100*samp_num), *args)
    print('Maximisation done '+str(max_x)+' '+str(max_y))
    samples = rej_sampling(pdf, args, low_l, up_l, max_y, samp_num)
    return samples
    
def test_pdf_2(x,c):
    return c*(-x**2 + 4*x)




# Choice with weights
'''
n = 50000
upp = 1000
arr = np.linspace(1/(10000*n), upp, 10000*upp)
prob = p(arr,0.5,20,3)
prob = prob/sum(prob)
samples = np.random.choice(arr, size=n, p=prob)
#plt.hist(samples, bins=1000)
'''


# Adapted from 1974 paper

def gamma_rej(a, n_samp, s=1):
    # For 0 < a <= 1
    # s is the same "scale" parameter of scipy.stats.gamma.pdf()
    # Easily used becuse if X ~ Gamma(a,theta),
    # then s*X ~ Gamma(a,s*theta)
    b = (np.e + a)/np.e
    samples = []
    while len(samples) < n_samp:
        u = random.random()
        p = u*b
        if p <= 1:
            x = p**(1/a)
            u_2 = random.random()
            if u_2 <= np.exp(-x):
                samples.append(s*x)
        else:
            x = -np.log((b-p)/a)
            u_2 = random.random()
            if u_2 <= x**(a-1):
                samples.append(s*x)
    return samples



# Adapted from 1974 paper with our modifications
# BB1_rej most simple only has a beta parameter,
# Unscaled_BB1_rej has lower shape parameter too
# Complete_BB1_rej samples from the full distribution, with beta, u and l parameters


def BB1_rej(beta, n_samp):
    a = beta + 1
    # For -1 < a <= 0
    b = (np.e + a + 1)/np.e
    samples = []
    while len(samples) < n_samp:
        u_1 = random.random()
        p = u_1*b
        if p <= 1:
            x = p**(1/(a+1))
            u_2 = random.random()
            if u_2 <= (1/x)*(1-np.exp(-x))*np.exp(-x):
                samples.append(x)
        else:
            x = -np.log((b-p)/(a+1))
            u_2 = random.random()
            if u_2 <= (1-np.exp(-x))*x**(a-1):
                samples.append(x)
    return samples


def Unscales_BB1_rej(beta, l, n_samp):
    a = beta + 1
    # For -1 < a <= 0
    b = (np.e + a + 1)/np.e
    samples = []
    while len(samples) < n_samp:
        u_1 = random.random()
        p = u_1*b
        if p <= 1:
            x = p**(1/(a+1))
            u_2 = random.random()
            if u_2 <= (l/x)*(1-np.exp(-x/l))*np.exp(-x):
                samples.append(x)
        else:
            x = -np.log((b-p)/(a+1))
            u_2 = random.random()
            if u_2 <= l*(1-np.exp(-x/l))*x**(a-1):
                samples.append(x)
    return samples


def Complete_BB1_rej(random_seed, beta, u, l, n_samp):
    # Developed For -2 < beta < -1
    # works for other beta values
    a = beta + 1
    l = l/u
    b = (np.e + a + 1)/np.e
    samples = []
    while len(samples) < n_samp:
        u_1 = random_seed.random()
        p = u_1*b
        if p <= 1:
            x = p**(1/(a+1))
            u_2 = random_seed.random()
            if u_2 <= (l/x)*(1-np.exp(-x/l))*np.exp(-x):
                samples.append(u*x)
        else:
            x = -np.log((b-p)/(a+1))
            u_2 = random_seed.random()
            if u_2 <= l*(1-np.exp(-x/l))*x**(a-1):
                samples.append(u*x)
    return samples



# BVMF Sampling (not sure this is the most appropriate file to develop these in)
# All designed for multiple samples even though we sample indiidually in the Event_Generator code
# Very efficient sampling, 100,000 samples < 1 sec for all
# In fact only the "von_misses_sampling" takes more than 0.1 sec for 1 million samples.

def burr_sampling(np_random_seed, c, k, l, n_samp):
    # simple Inverse Transform Sampling
    u_samp = np_random_seed.uniform(size = n_samp)
    samples = l*((1/((1-u_samp)**(1/k))) - 1)**(1/c)
    return samples

def von_misses_sampling(random_seed, mu, kappa, n_samp):
    # using ranodm module built-in function
    samples = []
    while len(samples) < n_samp:
        samp = random_seed.vonmisesvariate(mu, kappa)
        samples.append(samp)
    return np.array(samples)

def von_misses_fisher_sampling(np_random_seed, mu, kappa, n_samp):
    # produces n_samp samples around mu on the unit 2-sphere
    # based on "Numerically stable sampling of the von Mises
    # Fisher distribution on S-2 (and other tricks)
    # by Wenzel Jakob
    # Vecotrised for large sample sizes
    mu = mu/np.linalg.norm(mu)
    # Sampling V
    theta_samp = np_random_seed.uniform(0, 2*np.pi, n_samp)
    v_samp = np.array([np.cos(theta_samp), np.sin(theta_samp)])
    v_samp = v_samp.T
    # Sampling W
    u_samp = np_random_seed.uniform(size = n_samp)
    # w_samp = (1/kappa)*np.log(np.exp(-kappa) + 2*u_samp*np.sinh(kappa))
    w_samp = 1 + (1/kappa)*np.log(u_samp + (1-u_samp)*np.exp(-2*kappa))
    factor = np.sqrt(1 - w_samp**2)
    init_sample = np.hstack((v_samp * factor[:, None], w_samp.reshape(v_samp.shape[0], 1)))
    b1 = mu[0]
    b2 = mu[1]
    b3 = mu[2]
    # Using derivation based on Rodrigues' formula,
    # but result specific to the sampling starting mean vector of (0,0,1)
    rotation_matrix = np.array([[1 - (b1**2)/(1+b3) , -b1*b2, b1],
                                [-b1*b2 , 1 - (b2**2)/(1+b3) , b2],
                                [-b1 , -b2 , 1 - (b1**2 + b2**2)/(1+b3)]])
    final_sample = np.matmul(rotation_matrix, init_sample.T).T
    return final_sample





# %%
