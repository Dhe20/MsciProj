from Components.Universe import Universe
import numpy as np
import matplotlib.pyplot as plt

U = Universe(characteristic_luminosity=1, total_luminosity=100, seed = 42)

# def H_0_inference_3d_gamma():
#     gamma_marginalised = np.zeros(len(self.H_0_pdf))
#     expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
#     for H_0_index, H_0 in enumerate(self.H_0_range):
#         N1 = self.calc_N1(H_0)
#         expected_event_num_divded_by_gamma[H_0_index] = N1
#         scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
#         scaled_gamma_lower_lim =  self.SurveyAndEventData.gamma_lower_lim*N1
#         Nhat = self.SurveyAndEventData.detected_event_count
#         gamma_marginalised[H_0_index] = gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim)
#     self.gamma_marginalised = gamma_marginalised
#     self.expected_event_num_divded_by_gamma = expected_event_num_divded_by_gamma
#     return self.gamma_marginalised

BVM_c = 15
BVM_k = 2
BVM_kappa = 200
max_D = U.max_D
def burr_cdf(lam, max_D = U.max_D):
    c = 15
    k = 2
    cdf = 1 - (1 + (max_D / lam) ** c) ** -k
    return cdf
def calc_N1(H_0):
    global U
    N1 = 0
    for g_i, flux in enumerate(U.fluxes):
        D_gi = U.detected_redshifts[g_i]/H_0
        if U.dimension == 2:
            luminosity = 2*np.pi*flux*D_gi
        else:
            luminosity = 4*np.pi*flux*D_gi**2
        N1 += luminosity*burr_cdf(lam = D_gi)
    return N1

H_0_min = 10
H_0_max = 200
H_0s = np.arange(H_0_min,H_0_max)
min_d = np.min(U.detected_redshifts / H_0_max)
max_d = np.max(U.detected_redshifts / H_0_min)

N1s = []
distances = []
for H_0 in H_0s:
    distances.append(np.mean(U.detected_redshifts / H_0))
    N1s.append(calc_N1(H_0))
# plt.plot(H_0s, distances)
plt.plot(H_0s, N1s)

plt.ylabel("Coefficient Proportional to Expected Event Number")
plt.xlabel("H0")

# burr_x = np.linspace(0, 2*max_d, 1000)
# lams = np.linspace(min_d, max_d, 10)
# for lam in lams:
#     burr_y = []
#     for x in burr_x:
#         burr_y.append(burr_cdf(lam = lam, max_D = x))
#     plt.plot(burr_x,burr_y, label = str(lam))
# plt.legend()
# plt.show()

plt.plot()

