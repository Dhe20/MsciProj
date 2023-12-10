import powerbox as pbox
import numpy as np

power = lambda k: cluster_coeff * k ** -3

lnpb = pbox.LogNormalPowerBox(
    N=512,  # Number of grid-points in the box
    dim=dimension,  # 2D box
    pk=power,  # The power-spectrum
    boxlength=1.,  # Size of the box (sets the units of k in pk)
    seed=seed  # Set a seed to ensure the box looks the same every time (optional)
)

num_of_galaxies = n
clustered_sample = lnpb.create_discrete_sample(nbar=int(1.3 * num_of_galaxies),
                                               randomise_in_cell=True,  # nbar specifies the number density
                                               # min_at_zero=False  # by default the samples are centred at 0. This shifts them to be positive.
                                               )
index_of_galaxies = list(np.arange(0, len(clustered_sample), 1))
selected_index = random.sample(index_of_galaxies, k=num_of_galaxies)
selected_galaxies = clustered_sample[selected_index, :] * self.size * 2