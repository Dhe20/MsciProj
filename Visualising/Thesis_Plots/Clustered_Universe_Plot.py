##


import powerbox as pbox
import matplotlib.pyplot as plt
import random
import numpy as np
import powerbox as pb
import matplotlib
cluster_coeff=0.05
plt.style.use("default")

import time

start_time = time.perf_counter()

fig, ax = plt.subplots(3, 3, figsize = (8,8), constrained_layout=True)

strings = []
for i in range(-4, 1):
    if i ==0:
        strings.extend([r'$1$'])
    # elif i==-1:
    #     strings.extend([r'$10^{}$'.format("{" + str(i) + "}"), '1'])
    else:
        strings.extend([r'$10^{}$'.format("{" + str(i) + "}"), r'$5 \times 10^{}$'.format("{" + str(i) + "}")])
print(strings)

for i, coeff in enumerate([0.0001, 0.0005, 0.001, 0.005 ,0.01,0.05, 0.1,0.5,1]):

    power = lambda k: coeff * k ** -1.2

    lnpb = pbox.LogNormalPowerBox(
        N=256,                     # Number of grid-points in the box
        dim=2,                     # 2D box
        pk = power, # The power-spectrum
        boxlength = 1.0,           # Size of the box (sets the units of k in pk)
        seed = 5,
        # ensure_physical=True
        # Set a seed to ensure the box looks the same every time (optional)
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("Elapsed time: ", elapsed_time)
    y = lnpb.delta_x() - np.mean(lnpb.delta_x()) + 1
    cax1 = ax[i // 3, i % 3].imshow(y, extent = (-.5,.5, -.5, .5), norm=matplotlib.colors.LogNorm() , cmap = 'binary') #vmax = 10
    ax[i // 3, i % 3].set_title(r"$k_{0} = $" + strings[i], size = 20)
    ax[i // 3, i % 3].set_xticklabels([])
    ax[i // 3, i % 3].set_yticklabels([])

cbar = fig.colorbar(cax1, ax=ax, orientation='horizontal', shrink=0.8, aspect=40)
cbar.set_label(label = r'$\delta(x)$',size=15,weight='bold')
cbar.ax.tick_params(labelsize=20)

# image_format = 'pdf' # e.g .png, .svg, etc.
# image_name = 'Clustered_Universe_Plot' + "." +image_format
# plt.savefig("HighRes/"+image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=1200)
# plt.savefig("LowRes/"+image_name, format=image_format,  bbox_inches='tight', pad_inches=0.5, dpi=120)
#
# plt.show()


