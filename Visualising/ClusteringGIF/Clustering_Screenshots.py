import matplotlib.pyplot as plt
import powerbox as pbox
import numpy as np
import random
import time
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

plt.style.use('default')

cluster_coeffs = np.arange(0,11)
# cluster_coeffs = [5]
print(cluster_coeffs)

for cluster_coeff in tqdm(cluster_coeffs):
    fig = plt.figure(figsize=(16, 8))

    # Define the grid layout
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.05, 1, 1])

    # Create the first subplot
    ax0 = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[0])
    ax = [ax0, ax1, cbar_ax]

# cluster_coeff = 60
    dimension = 2
    seed = 28


    power = lambda k: cluster_coeff * k ** -3

    lnpb = pbox.PowerBox(
        N=512,  # Number of grid-points in the box
        dim=dimension,  # 2D box
        pk=power,  # The power-spectrum
        boxlength=1.,  # Size of the box (sets the units of k in pk)
        seed=seed,  # Set a seed to ensure the box looks the same every time (optional)
        ensure_physical = True
    )



    num_of_galaxies = 10000
    clustered_sample = lnpb.create_discrete_sample(nbar=int(1.3 * num_of_galaxies),
                                                   randomise_in_cell=True,  # nbar specifies the number density
                                                   # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                                   )
    index_of_galaxies = list(np.arange(0, len(clustered_sample), 1))
    random.seed(seed)
    selected_index = random.sample(index_of_galaxies, k=num_of_galaxies)
    selected_galaxies = clustered_sample[selected_index]

    ax[1].scatter(selected_galaxies[:,1],-selected_galaxies[:,0], alpha = 0.3, marker = ".", color = 'black')
    ax[1].set_aspect('equal', 'box')  # Make the second subplot square
    ax[1].set_adjustable('box')
    ax[1].set_facecolor('white')
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Adjust layout to prevent overlapping



    suptitle_text = fig.suptitle(r'$\xi_{0} = $' + str(cluster_coeff), fontsize=30, color='black', va='top')
    ax[0].set_title('Over-Density Heatmap', pad=10, fontsize=30, color='black')
    ax[1].set_title('Sampled Galactic Survey', pad=10, fontsize=30, color='black')

    # Display the figure
    fig.subplots_adjust(top=0.85)

    # Calculate the center x position between the first two subplots
    left_ax_pos = ax[0].get_position()
    middle_ax_pos = ax[1].get_position()
    center_x = (left_ax_pos.x1 + middle_ax_pos.x0) / 2

    # Update the suptitle position
    suptitle_text.set_position((center_x, left_ax_pos.y1 + 0.15))



    y = lnpb.delta_x()

    im = ax[0].imshow(lnpb.delta_x(), extent = (-.5,.5, -.5, .5), vmin = -1)
    ax[0].set_aspect('equal', 'box')  # Make the first subplot square
    ax[0].set_adjustable('box')

    ax[2].set_aspect(1.95)
    # Create the colorbar in the newly created axis
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.set_ticks_position('left')
    # plt.show()
    plt.savefig('/Users/daneverett/PycharmProjects/MSciProject/Visualising/ClusteringGIF/Images/figure_cluster_coeff_{}.png'.format(cluster_coeff), facecolor='white')
    plt.close(fig)

    plt.tight_layout()

# ax[i % 3, i // 3].set_title("Cluster Coefficient: " + str(round(coeff,2)))
# ax[i % 3, i // 3].set_xticklabels([])
# ax[i % 3, i // 3].set_yticklabels([])

# fig.colorbar(cax1, ax=ax, orientation='vertical', shrink=0.8, aspect=40)



# num_of_galaxies = 1000
# clustered_sample = lnpb.create_discrete_sample(nbar=int(1.3 * num_of_galaxies),
#                                                randomise_in_cell=True,  # nbar specifies the number density
                                               # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                               # )

# index_of_galaxies = list(np.arange(0, len(clustered_sample), 1))
# selected_index = random.sample(index_of_galaxies, k=num_of_galaxies)
# selected_galaxies = clustered_sample[selected_index, :] * self.size * 2