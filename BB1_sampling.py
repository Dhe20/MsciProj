#%%
import numpy as np
import scipy.stats as ss
from math import gamma, modf

#%%
x = np.linspace(0.0, 10.0, 10000000)

def neg_gamma(z):
    # Full analytic continuation of gamma function
    # to negative axis - uses Recurrence formula
    n = - modf(z)[1]+1
    p_l = np.arange(z,z+n+1)
    g = gamma(z+n+1)/np.prod(p_l)
    return g


def p(x,b,u,l):
    b = b-2
    #print(b)
    if b>-2 and b!=-1:
    #if (b>-2).all() and (b!=-1).all():
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
    elif b==-1:
    #elif (b==-1).all():
        C = 1/(np.log(1+u/l))
    #print(C)
    p =(C/u)*(1-np.exp(-x/l))*((x/u)**b)*np.exp(-x/u)
    return p

# Define the distribution using rv_continuous
class Full_Schechter(ss.rv_continuous): 
    def _pdf(self, x, b,u,l):
        return p(x,b,u,l)#(1.0/const) * p(x)

BB1 = Full_Schechter(name="BB1", a=0.0)

b = 0.5
u = 5
l = 2

# create pdf, cdf, random samples
#pdf = BB1.pdf(x = x, b=b,u=u,l=l,c=c)
#cdf = BB1.cdf(x = x, b=b,u=u,l=l,c=c)
samples = BB1.rvs(b=b,u=u,l=l, size = 1000)

#%%