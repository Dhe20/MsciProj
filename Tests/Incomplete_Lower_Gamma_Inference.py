from scipy.special import gammainc
import numpy as np

# upper_lim = 10**12
# lower_lim = 10
# Nhat = 5.5
#
# print(gammainc(Nhat, upper_lim)-gammainc(Nhat, lower_lim))

seed = 1
np_rand_state = np.random.default_rng(seed)
event_count = np_rand_state.poisson(lam = 5)
