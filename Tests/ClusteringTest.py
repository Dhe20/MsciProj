##


import powerbox as pbox
import matplotlib.pyplot as plt
import random
import numpy as np
import powerbox as pb
cluster_coeff=0.05

import time

start_time = time.perf_counter()

fig, ax = plt.subplots(3, 2)

for i, coeff in enumerate([0.1,5,10,20,30,50]):

    power = lambda k: coeff * k ** -3

    lnpb = pbox.LogNormalPowerBox(
        N=256,                     # Number of grid-points in the box
        dim=2,                     # 2D box
        pk = power, # The power-spectrum
        boxlength = 1.0,           # Size of the box (sets the units of k in pk)
        seed = 42                # Set a seed to ensure the box looks the same every time (optional)
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Elapsed time: ", elapsed_time)
    y = lnpb.delta_x()
    cax1 = ax[i%3,i//3].imshow(lnpb.delta_x(), extent = (-.5,.5, -.5, .5), vmin = -1, vmax = 8.99385236638655)
    ax[i % 3, i // 3].set_title("Cluster Coefficient: " + str(round(coeff,2)))
    ax[i % 3, i // 3].set_xticklabels([])
    ax[i % 3, i // 3].set_yticklabels([])

fig.colorbar(cax1, ax=ax, orientation='vertical', shrink=0.8, aspect=40)


# fig.constrained_layout()
plt.show()
    # time.sleep(2)

##

power = lambda k: 0.01 * k ** -3
start_time = time.perf_counter()
lnpb = pbox.LogNormalPowerBox(
    N=256,  # Number of grid-points in the box
    dim=3,  # 2D box
    pk=power,  # The power-spectrum
    boxlength=1.0,  # Size of the box (sets the units of k in pk)
    seed=42  # Set a seed to ensure the box looks the same every time (optional)
)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)


start_time = time.perf_counter()
N=5000
ClusteredSample = lnpb.create_discrete_sample(nbar=int(2*N),
                                              randomise_in_cell= True,
                                              # nbar specifies the number density
                                    # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                   )
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

##

p,k = pbox.get_power(lnpb.delta_x(),lnpb.boxlength)
plt.plot(k,p)
plt.xscale('log')
plt.yscale('log')

plt.show()
##
N = 1000

start_time = time.perf_counter()


ClusteredSample = lnpb.create_discrete_sample(nbar=int(2*N),
                                              randomise_in_cell= True,
                                              # nbar specifies the number density
                                    # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                   )

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)


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