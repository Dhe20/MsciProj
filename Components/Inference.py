#%%
import numpy as np
import scipy as sp
from scipy.integrate import quad
from Components.SurveyAndEventData import SurveyAndEventData
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import gammainc

#%%

class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData, gamma = True, vectorised = True,
                 survey_type='perfect', resolution_H_0=100,
                 H_0_Min = 50, H_0_Max = 100):
        self.SurveyAndEventData = SurveyAndEventData
        self.distribution_calculated = False
        self.H_0_Min = H_0_Min
        self.H_0_Max = H_0_Max
        self.survey_type = survey_type
        self.gamma = gamma
        self.vectorised = vectorised
        self.resolution_H_0 = resolution_H_0
        self.H_0_pdf = np.zeros(self.resolution_H_0)
        self.H_0_range = np.linspace(self.H_0_Min, self.H_0_Max, self.resolution_H_0)
        self.H_0_increment = self.H_0_range[1] - self.H_0_range[0]
        self.H_0_pdf_single_event = np.zeros(shape = (len(self.SurveyAndEventData.BH_detected_coords), self.resolution_H_0))

        self.inference_method = dict({"perfect2d":self.H_0_inference_2d_perfect_survey,
                                "imperfect2d": self.H_0_inference_2d_imperfect_survey,
                                "perfect3d": self.H_0_inference_3d_perfect_survey,
                                "imperfect3d": self.H_0_inference_3d_imperfect_survey,
                                "perfectvectorised3d": self.H_0_inference_3d_perfect_survey_vectorised
        })
        self.gamma_method = dict({
            "2d": self.H_0_inference_2d_gamma,
            "3d": self.H_0_inference_3d_gamma
        })

        if self.SurveyAndEventData.dimension==3 and self.survey_type == "perfect" and self.vectorised:
            self.inference_method_name = self.survey_type + "vectorised" + str(self.SurveyAndEventData.dimension) + "d"
        else:
            self.inference_method_name = self.survey_type + str(self.SurveyAndEventData.dimension) + "d"

        self.g_H_0 = dict()

        self.countour = dict({"gauss": self.gauss_p_hat_g_true,
                              "BVM": self.BVM_p_hat_g_true})


    def H_0_Prob(self):
        self.distribution_calculated = True
        self.H_0_pdf = self.inference_method[self.inference_method_name]()
        if self.gamma:
            gamma_marginalisation = self.gamma_method[str(self.SurveyAndEventData.dimension)+"d"]()
            self.H_0_pdf *= gamma_marginalisation
        return self.H_0_pdf


    def BVM_p_hat_g_true(self, dim, D, u_r, u_phi, u_theta=0, phi=0, theta=0):
        if dim == 2:
            p = self.SurveyAndEventData.burr(u_r,
                self.SurveyAndEventData.BVM_c,
                self.SurveyAndEventData.BVM_k,
                D) *self.SurveyAndEventData.von_misses(u_phi,
                phi, self.SurveyAndEventData.BVM_kappa) * u_r
        elif dim == 3:
            p = self.SurveyAndEventData.burr(u_r,
                self.SurveyAndEventData.BVM_c,
                self.SurveyAndEventData.BVM_k,
                D) * self.SurveyAndEventData.von_misses_fisher(
                phi, theta, u_phi, u_theta, self.SurveyAndEventData.BVM_kappa) * (u_r**2)*np.sin(u_theta)
        return p

    def gauss_p_hat_g_true(self, dim, D, u_r, u_phi, u_theta=0, phi=0, theta=0):
        if dim == 2:
            sig = self.SurveyAndEventData.noise_sigma
            x = D**2 + u_r**2 - 2*u_r*D*(np.cos(u_phi-phi))
            p = u_r * (1/(2*np.pi*sig**2))*np.exp(-x/(2*sig**2))

        elif dim == 3:
            sig = self.SurveyAndEventData.noise_sigma
            x = D**2 + u_r**2 - 2*u_r*D*(np.sin(theta) * np.sin(u_theta) * np.cos(u_phi-phi) + np.cos(theta) * np.cos(u_theta))
            p = (u_r**2) * np.sin(u_theta) * (1/((2*np.pi*sig**2)**(3/2)))*np.exp(-x/(2*sig**2))
        return p

    def H_0_inference_2d_perfect_survey(self):
        for event_num in tqdm(range(len(self.SurveyAndEventData.BH_detected_coords))):
            H_0_pdf_single_event = np.zeros(self.resolution_H_0)
            u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords[event_num])))
            u_x = self.SurveyAndEventData.BH_detected_coords[event_num][0]
            u_y = self.SurveyAndEventData.BH_detected_coords[event_num][1]
            u_phi = np.arctan2(u_y, u_x)
            for H_0_index, H_0 in enumerate(self.H_0_range):
                g_H_0_slice = []
                H_0_pdf_slice_single_event = 0
                for g in (range(len(self.SurveyAndEventData.detected_luminsoties))):
                    X = self.SurveyAndEventData.detected_coords[g][0]
                    Y = self.SurveyAndEventData.detected_coords[g][1]
                    phi = np.arctan2(Y, X)
                    D = (self.SurveyAndEventData.detected_redshifts[g]) / H_0
                    galaxy_H_0_contribution = (D * self.SurveyAndEventData.fluxes[g] * u_r *
                                                   self.countour[self.SurveyAndEventData.noise_distribution](
                                                       self.SurveyAndEventData.dimension, D, u_r, u_phi, phi=phi))
                    g_H_0_slice.append(galaxy_H_0_contribution)
                    H_0_pdf_slice_single_event += galaxy_H_0_contribution
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event
                self.g_H_0[str(H_0)] = g_H_0_slice
            self.H_0_pdf_single_event[event_num] = H_0_pdf_single_event #/ (
                       # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            else:
                self.H_0_pdf *= H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey(self):
        for event_num in tqdm(range(len(self.SurveyAndEventData.BH_detected_coords))):
            H_0_pdf_single_event = np.zeros(self.resolution_H_0)
            u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords[event_num])))
            u_x = self.SurveyAndEventData.BH_detected_coords[event_num][0]
            u_y = self.SurveyAndEventData.BH_detected_coords[event_num][1]
            u_z = self.SurveyAndEventData.BH_detected_coords[event_num][2]
            u_phi = np.arctan2(u_y, u_x)
            u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)
            for H_0_index, H_0 in enumerate(self.H_0_range):
                g_H_0_slice = []
                H_0_pdf_slice_single_event = 0
                for g in (range(len(self.SurveyAndEventData.detected_luminsoties))):
                    X = self.SurveyAndEventData.detected_coords[g][0]
                    Y = self.SurveyAndEventData.detected_coords[g][1]
                    Z = self.SurveyAndEventData.detected_coords[g][2]
                    phi = np.arctan2(Y, X)
                    XY = np.sqrt((X) ** 2 + (Y) ** 2)
                    theta = np.arctan2(XY, Z)
                    D = (self.SurveyAndEventData.detected_redshifts[g]) / H_0
                    galaxy_H_0_contribution = ((D**2) * self.SurveyAndEventData.fluxes[g]
                                                * self.countour[self.SurveyAndEventData.noise_distribution]
                                (self.SurveyAndEventData.dimension, D, u_r, u_phi, u_theta=u_theta, phi=phi, theta=theta))

                    g_H_0_slice.append(galaxy_H_0_contribution)
                    H_0_pdf_slice_single_event += galaxy_H_0_contribution
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event
                self.g_H_0[str(H_0)] = g_H_0_slice
            self.H_0_pdf_single_event[event_num] = H_0_pdf_single_event #/ (
                       # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            else:
                self.H_0_pdf *= H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = redshifts * H_0_recip

        burr_full = self.get_vectorised_burr(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = np.square(Ds) * self.SurveyAndEventData.fluxes

        full_expression = burr_full * vmf * luminosity_term
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def p_D_prior(self, D, z_hat, sigma_z, H_0, u_r):
        sigma_D = sigma_z/H_0
        N = (1/np.sqrt(2*np.pi*(sigma_D**2)))*np.exp(-0.5*((D - z_hat/H_0)/sigma_D)**2)
        p = N * self.SurveyAndEventData.burr(u_r,self.SurveyAndEventData.BVM_c,self.SurveyAndEventData.BVM_k,D)
        return p

    def H_0_inference_2d_imperfect_survey(self):
        for event_num in tqdm(range(len(self.SurveyAndEventData.BH_detected_coords))):
            H_0_pdf_single_event = np.zeros(self.resolution_H_0)
            for H_0_index, H_0 in enumerate(self.H_0_range):
                H_0_pdf_slice_single_event = 0
                u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords[event_num])))
                u_x = self.SurveyAndEventData.BH_detected_coords[event_num][0]
                u_y = self.SurveyAndEventData.BH_detected_coords[event_num][1]
                u_phi = np.arctan2(u_y, u_x)
                for g in (range(len(self.SurveyAndEventData.detected_luminsoties))):
                    X = self.SurveyAndEventData.detected_coords[g][0]
                    Y = self.SurveyAndEventData.detected_coords[g][1]
                    phi = np.arctan2(Y, X)
                    D = (self.SurveyAndEventData.detected_redshifts[g]) / H_0
                    partial_int = lambda z_har, sigma_z, H_0, u_r: quad(self.p_D_prior, 0, np.inf, args=(z_har, sigma_z, H_0, u_r,))
                    sigma_z = self.SurveyAndEventData.detected_redshifts_uncertainties[g]
                    z_hat = self.SurveyAndEventData.detected_redshifts[g]
                    I = partial_int(z_hat, sigma_z, H_0, u_r)
                    H_0_pdf_slice_single_event += D * self.SurveyAndEventData.fluxes[g]* I[0] * u_r * self.SurveyAndEventData.von_misses(u_phi,
                                                                                    phi, self.SurveyAndEventData.BVM_kappa)
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event

            self.H_0_pdf_single_event[event_num] = H_0_pdf_single_event  # / (
            # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event  # /(
                # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            else:
                self.H_0_pdf *= H_0_pdf_single_event  # /(
                # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def H_0_inference_3d_imperfect_survey(self):
        for event_num in tqdm(range(len(self.SurveyAndEventData.BH_detected_coords))):
            H_0_pdf_single_event = np.zeros(self.resolution_H_0)
            for H_0_index, H_0 in enumerate(self.H_0_range):
                H_0_pdf_slice_single_event = 0
                u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords[event_num])))
                u_x = self.SurveyAndEventData.BH_detected_coords[event_num][0]
                u_y = self.SurveyAndEventData.BH_detected_coords[event_num][1]
                u_z = self.SurveyAndEventData.BH_detected_coords[event_num][2]
                u_phi = np.arctan2(u_y, u_x)
                u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)
                for g in (range(len(self.SurveyAndEventData.detected_luminsoties))):
                    X = self.SurveyAndEventData.detected_coords[g][0]
                    Y = self.SurveyAndEventData.detected_coords[g][1]
                    Z = self.SurveyAndEventData.detected_coords[g][2]
                    phi = np.arctan2(Y, X)
                    XY = np.sqrt((X) ** 2 + (Y) ** 2)
                    theta = np.arctan2(XY, Z)
                    D = (self.SurveyAndEventData.detected_redshifts[g]) / H_0
                    partial_int = lambda z_har, sigma_z, H_0, u_r: quad(self.p_D_prior, 0, np.inf, args=(z_har, sigma_z, H_0, u_r,))
                    sigma_z = self.SurveyAndEventData.detected_redshifts_uncertainties[g]
                    z_har = self.SurveyAndEventData.detected_redshifts[g]
                    I = partial_int(z_har, sigma_z, H_0, u_r)
                    H_0_pdf_slice_single_event += (D**2) * self.SurveyAndEventData.fluxes[g] *(u_r**2) * np.sin(u_theta) * I[0] * self.SurveyAndEventData.von_misses_fisher(
                                                                                    u_phi, phi, u_theta, theta, self.SurveyAndEventData.BVM_kappa)
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event
            self.H_0_pdf_single_event[event_num] = H_0_pdf_single_event #/ (
                       # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
            else:
                self.H_0_pdf *= H_0_pdf_single_event#/(
                        # np.sum(H_0_pdf_single_event) * (self.H_0_increment))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf


    def H_0_inference_2d_gamma(self):
        gamma_marginalised = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1(H_0)
            scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
            scaled_gamma_lower_lim =  self.SurveyAndEventData.gamma_lower_lim*N1
            Nhat = self.SurveyAndEventData.detected_event_count
            gamma_marginalised[H_0_index] = gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim)
        self.gamma_marginalised = gamma_marginalised
        return self.gamma_marginalised

    def H_0_inference_3d_gamma(self):
        gamma_marginalised = np.zeros(len(self.H_0_pdf))
        expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1(H_0)
            expected_event_num_divded_by_gamma[H_0_index] = N1
            scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
            scaled_gamma_lower_lim =  self.SurveyAndEventData.gamma_lower_lim*N1
            Nhat = self.SurveyAndEventData.detected_event_count
            gamma_marginalised[H_0_index] = gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim)
        self.gamma_marginalised = gamma_marginalised
        self.expected_event_num_divded_by_gamma = expected_event_num_divded_by_gamma
        return self.gamma_marginalised

    def plot_H_0(self):
        if not self.distribution_calculated:
            self.H_0_Prob()
        plt.plot(self.H_0_range, self.H_0_pdf/(np.sum(self.H_0_pdf)*self.H_0_increment))
        plt.axvline(x=70, c='r', ls='--')
        plt.show()

    def H_0_posterior(self):
        self.H_0_Prob()
        p = self.H_0_pdf/np.sum(self.H_0_pdf)
        x = self.H_0_range
        return [x, p]

    def burr_cdf(self, lam):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        max_D = self.SurveyAndEventData.max_D
        cdf = 1-(1 + (max_D/lam)**c)**-k
        return cdf

    def get_vectorised_burr(self, Ds):
        Ds_tile = np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1))

        recip_Ds_tile = np.reciprocal(Ds_tile)

        u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))[:, np.newaxis, np.newaxis]

        omegas = recip_Ds_tile * u_r

        burr_term1 = np.power(omegas, self.SurveyAndEventData.BVM_c - 1)
        burr_term2 = np.power(1 + np.power(omegas, self.SurveyAndEventData.BVM_c), - self.SurveyAndEventData.BVM_k - 1)

        burr_full = self.SurveyAndEventData.BVM_k * self.SurveyAndEventData.BVM_c * recip_Ds_tile * burr_term1 * burr_term2
        return burr_full
    def get_vectorised_vmf(self):
        kappa = self.SurveyAndEventData.BVM_kappa
        vmf_C = kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))

        u_x = self.SurveyAndEventData.BH_detected_coords[:, 0]
        u_y = self.SurveyAndEventData.BH_detected_coords[:, 1]
        u_z = self.SurveyAndEventData.BH_detected_coords[:, 2]

        u_phi = np.arctan2(u_y, u_x)[:, np.newaxis]
        u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)[:, np.newaxis]

        X = self.SurveyAndEventData.detected_coords[:, 0]
        Y = self.SurveyAndEventData.detected_coords[:, 1]
        Z = self.SurveyAndEventData.detected_coords[:, 2]
        XY = np.sqrt((X) ** 2 + (Y) ** 2)

        phi = np.tile(np.arctan2(Y, X), (self.SurveyAndEventData.detected_event_count, 1))
        theta = np.tile(np.arctan2(XY, Z), (self.SurveyAndEventData.detected_event_count, 1))

        sin_u_theta = np.sin(u_theta)
        cos_u_theta = np.cos(u_theta)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        cos_phi_diff = np.cos(phi - u_phi)

        vmf = vmf_C * np.exp(kappa * (sin_theta * sin_u_theta * cos_phi_diff
                                      + cos_theta * cos_u_theta))[:, np.newaxis,:]
        return vmf

    def calc_N1(self, H_0):
        N1 = 0
        for g_i, flux in enumerate(self.SurveyAndEventData.fluxes):
            D_gi = self.SurveyAndEventData.detected_redshifts[g_i]/H_0
            if self.SurveyAndEventData.dimension == 2:
                luminosity = 2*np.pi*flux*D_gi
            else:
                luminosity = 4*np.pi*flux*D_gi**2
            N1 += luminosity*self.burr_cdf(lam = D_gi)
        N1 *= self.SurveyAndEventData.sample_time
        return N1



from Components.EventGenerator import EventGenerator
# # '''
# Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.01, event_rate=10,
#                         luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
#                         cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=1,
#                         event_distribution="Proportional", noise_distribution = "BVM", contour_type = "BVM", redshift_noise_sigma = 0.0,
#                         resolution=500, plot_contours = False, seed = 1)

# Gen.plot_universe_and_events()
# Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data, survey_type='gamma')
#
# CDF = []
# X = np.arange(1,100)
# for elem in X:
#     CDF.append(Y.burr_cdf(elem))
#
# plt.plot(X, CDF)
# plt.yscale("log")
# Y2 = Inference(Data, survey_type='perfect')
# plt.plot(Y.H_0_range, Y.H_0_Prob())
# plt.plot(Y.H_0_range, Y2.H_0_Prob())
# plt.show()
#
# plt.plot(Y.H_0_range, Y.gamma_marginalised)
# plt.show()
#
#
# '''




# %%
