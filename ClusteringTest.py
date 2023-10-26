import powerbox as pbox
import matplotlib.pyplot as plt
import random
import numpy as np

power = lambda k: 20*k**-3

lnpb = pbox.LogNormalPowerBox(
    N=512,                     # Number of grid-points in the box
    dim=2,                     # 2D box
    pk = power, # The power-spectrum
    boxlength = 1.0,           # Size of the box (sets the units of k in pk)
    seed = 513                # Set a seed to ensure the box looks the same every time (optional)
)
# y = lnpb.delta_x()
plt.imshow(lnpb.delta_x(), extent = (-.5,.5, -.5, .5))
plt.colorbar()
plt.show()


##

p,k = pbox.get_power(lnpb.delta_x(),lnpb.boxlength)
plt.plot(k,p)
plt.xscale('log')
plt.yscale('log')

plt.show()
##
N = 1000000
ClusteredSample = lnpb.create_discrete_sample(nbar=int(2*N),
                                              randomise_in_cell= True,# nbar specifies the number density
                                    # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                   )
NLengthList = list(np.arange(0, len(ClusteredSample), 1))
Chosen = random.sample(NLengthList, k = N)
ClusteredSample1 = ClusteredSample[0:N]
ClusteredSample2 = ClusteredSample[N:2*N]
ClusteredSample3 = ClusteredSample[Chosen,:]

x0 = ClusteredSample[:,1]
y0 = -ClusteredSample[:,0]

x = ClusteredSample1[:,1]
y = -ClusteredSample1[:,0]

x2 = ClusteredSample2[:,1]
y2= -ClusteredSample2[:,0]

x3 = ClusteredSample3[:,1]
y3 = -ClusteredSample3[:,0]

# plt.scatter(x, y, alpha=1,s=2)
# plt.show()
##
h = plt.hist2d(x, y, bins = 100)
plt.colorbar(h[3])
plt.show()

##
h = plt.hist2d(x2, y2, bins = 100)
plt.colorbar(h[3])
plt.show()


##

h = plt.hist2d(x0, y0, bins = 100)
plt.colorbar(h[3])
plt.show()

##

h = plt.hist2d(x3, y3, bins = 100)
plt.colorbar(h[3])
plt.show()

##
# import matplotlib.pyplot as plt
# import scipy.special as sps
# import numpy as np
#
# shape, scale = 2, .3  # mean=4, std=2*sqrt(2)
# s = np.random.gamma(shape, scale, 1000)
# count, bins, ignored = plt.hist(s, 50, density=True)
# y = bins**(shape-1)*(np.exp(-bins/scale) /
#                      (sps.gamma(shape)*scale**shape))
#
# plt.plot(bins, y, linewidth=2, color='r')
# plt.show()
#
# print(np.min(s))