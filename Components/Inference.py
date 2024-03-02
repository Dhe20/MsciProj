#%%
import numpy as np
import math
import scipy as sp
import scipy.stats as sps
from scipy.integrate import quad, dblquad
from Components.SurveyAndEventData import SurveyAndEventData
from Tools.BB1_sampling import p as BB1_pdf, neg_gamma
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import gammaincc
from scipy.special import gamma as gamma_function
from mpmath import gammainc as mp_gammainc
import scipy.stats as ss
import time



class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData, gamma = True, vectorised = True, event_distribution_inf='Proportional', lum_function_inf = 'Full-Schechter', gauss=False, p_det=True, poster=False,
                 survey_type='perfect', resolution_H_0=100, H_0_Min = 50, H_0_Max = 100, gamma_known = False, gauss_type = "Cartesian"):

        self.SurveyAndEventData = SurveyAndEventData
        self.c = self.SurveyAndEventData.c


        self.distribution_calculated = False
        self.H_0_Min = H_0_Min
        self.H_0_Max = H_0_Max
        self.survey_type = survey_type
        self.event_distribution_inf = event_distribution_inf
        self.lum_function_inf = lum_function_inf
        self.gamma = gamma
        self.gauss = gauss
        self.gauss_type = gauss_type
        self.p_det = p_det
        self.poster = poster
        self.vectorised = vectorised
        self.gamma_known = gamma_known
        self.resolution_H_0 = resolution_H_0
        self.H_0_pdf = np.zeros(self.resolution_H_0)
        self.H_0_range = np.linspace(self.H_0_Min, self.H_0_Max, self.resolution_H_0)
        self.H_0_increment = self.H_0_range[1] - self.H_0_range[0]
        self.H_0_pdf_single_event = np.zeros(shape = (len(self.SurveyAndEventData.BH_detected_coords), self.resolution_H_0))
        self.event_selection = self.SurveyAndEventData.event_distribution

        self.inference_method = dict({"perfect2d":self.H_0_inference_2d_perfect_survey,
                                # "imperfect2d": self.H_0_inference_2d_imperfect_survey,
                                # "perfect3d": self.H_0_inference_3d_perfect_survey,
                                # "imperfect3d": self.H_0_inference_3d_imperfect_survey,
                                "perfectvectorised2d": self.H_0_inference_2d_perfect_survey_vectorised,
                                "perfectvectorised3d": self.H_0_inference_3d_perfect_survey_vectorised,
                                "perfectvectorised2dGaussRadial": self.H_0_inference_2d_perfect_survey_vectorised_gaussian_radius,
                                "perfectvectorised3dGaussRadial": self.H_0_inference_3d_perfect_survey_vectorised_gaussian_radius,
                                "imperfectvectorised3d": self.H_0_inference_3d_imperfect_survey_vectorised,
                                "perfectvectorised3dGaussCartesian": self.H_0_inference_3d_perfect_survey_vectorised_gaussian,
                                "perfectvectorised3dposter": self.poster_H_0_inference_3d_perfect_survey_vectorised,
        })
        self.gamma_method = dict({
            "2d": self.H_0_inference_gamma, #Method is identical to 3d, no point writing it twice.
            "3d": self.H_0_inference_gamma,
            "2dGauss": self.H_0_inference_2d_gamma_gaussian,
            "3dGauss": self.H_0_inference_gamma_gaussian,
            "3dGammaKnown": self.H_0_inference_gamma_known
            #Need a function for cartesian Gaussian
        })
        #Fix the if statement for selecting Inference method
        
        self.lum_term = dict({'Random':self.get_lum_term_random, 
                              'Proportional': self.get_lum_term_proportional})

        self.lum_term_integrand = dict({'Random':self.var_lum_const, 
                              'Proportional': self.var_lum_prop})

        self.lum_function_integrand = dict({'Full-Schechter':self.var_BB1,
                                            'Shoddy-Schechter': self.var_schechter})



        #if self.SurveyAndEventData.dimension==3 and self.survey_type == "perfect" and self.vectorised:
        #    self.inference_method_name = self.survey_type + "vectorised" + str(self.SurveyAndEventData.dimension) + "d"
        
        if not self.gauss:
            self.inference_method_name = self.survey_type + "vectorised" + str(self.SurveyAndEventData.dimension) + "d"
        elif self.gauss:
            self.inference_method_name = self.survey_type + "vectorised" + str(
                self.SurveyAndEventData.dimension) + "d" + "Gauss" + self.gauss_type
        
        if self.poster:
            self.inference_method_name+='poster'
        
        self.g_H_0 = dict()

        self.countour = dict({"gauss": self.gauss_p_hat_g_true,
                              "BVM": self.BVM_p_hat_g_true})


    def H_0_Prob(self):
        self.distribution_calculated = True
        self.H_0_pdf = self.inference_method[self.inference_method_name]()
        if self.gamma: #This step is unvectorised but still only takes ~1/4 of the time of first inference stage
            gamma_method_name = str(self.SurveyAndEventData.dimension)+"d"
            if self.gauss:
                gamma_method_name += "Gaussian"
            elif self.gamma_known:
                gamma_method_name += "GammaKnown"
            gamma_marginalisation = self.gamma_method[gamma_method_name]()
            self.H_0_pdf *= gamma_marginalisation
        if not self.poster:  
            self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
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
                    D = (self.SurveyAndEventData.detected_redshifts[g]) * self.c / H_0
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
        if self.SurveyAndEventData.noise_distribution == "BVMF_eff":
            contour_type = "BVM"
        else:
            contour_type = self.SurveyAndEventData.noise_distribution
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
                    D = (self.SurveyAndEventData.detected_redshifts[g]) * self.c/ H_0
                    galaxy_H_0_contribution = ((D**2) * self.SurveyAndEventData.fluxes[g]
                                                * self.countour[contour_type]
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

    def get_lum_term_proportional(self, Ds):
        return np.square(Ds) * self.SurveyAndEventData.fluxes

    def get_lum_term_random(self, Ds):
        return 1

    def H_0_inference_2d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        burr_full = self.get_vectorised_burr(Ds)

        vm = self.get_vectorised_vm()

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = burr_full * vm * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)

        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            self.H_0_pdf = self.H_0_pdf/P_det_total_power
        
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = redshifts * H_0_recip * self.c

        burr_full = self.get_vectorised_burr(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = burr_full * vmf * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)

        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            self.H_0_pdf = self.H_0_pdf/P_det_total_power
        
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def poster_H_0_inference_3d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = redshifts * H_0_recip * self.c

        burr_full = self.get_vectorised_burr(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = burr_full * vmf * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        #self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)

        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            #P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            #self.H_0_pdf = self.H_0_pdf/P_det_total_power
            self.H_0_pdf_single_event = self.H_0_pdf_single_event/P_det_total
        
        self.H_0_pdf_single_event /= np.sum(self.H_0_pdf_single_event, axis=1)[:,np.newaxis] * (self.H_0_increment)
        return self.H_0_pdf_single_event

    def get_p_det_vec(self, Ds):
        threshold = self.SurveyAndEventData.max_D
        if self.gauss:
            z = (threshold - Ds) / (np.sqrt(2) * self.SurveyAndEventData.noise_sigma)
            # should be adjusted to have std proportional to Ds
            p = 0.5 * (1 + sp.special.erf(z))
        else:
            d_inv = np.reciprocal(Ds)
            arg = threshold * d_inv
            term_1 = np.power(arg, self.SurveyAndEventData.BVM_c)
            term_2 = np.power(1 + term_1, -self.SurveyAndEventData.BVM_k)
            p = 1 - term_2
        return p

    def H_0_inference_3d_imperfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        # For now I am keeping constand sigma_z
        redshifts_uncertanties = np.tile(np.full(len(self.SurveyAndEventData.detected_redshifts),
                                                 self.SurveyAndEventData.redshift_noise_sigma),
                                         (self.resolution_H_0, 1))

        us = redshifts * H_0_recip * self.c
        ss = redshifts_uncertanties * H_0_recip * self.c

        burr_full = self.get_vectorised_normal_burr_int(us, ss)

        # self.burr_full = burr_full

        vmf = self.get_vectorised_vmf()

        # luminosity_term = np.square(us) * self.SurveyAndEventData.fluxes

        luminosity_term = self.lum_term[self.event_distribution_inf](us)

        full_expression = burr_full * vmf * luminosity_term

        # self.full = full_expression

        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        # self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        # self.H_0_pdf = np.exp(self.log_H_0_pdf - np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)]))
        # self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(us)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            self.H_0_pdf_single_event = self.H_0_pdf_single_event / P_det_total

        f = np.vectorize(math.frexp)
        split = f(self.H_0_pdf_single_event)
        flo = split[0]
        ex = split[1]
        p_flo = np.prod(flo, axis=0)
        p_ex = np.sum(ex, axis=0)
        scaled_ex = p_ex - np.max(p_ex)
        scaled_flo = p_ex / p_flo[np.argmax(p_ex)]
        self.H_0_pdf = scaled_flo * (0.5 ** (-1 * scaled_ex))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey_vectorised_gaussian(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = redshifts * H_0_recip * self.c

        gauss_full = self.get_vectorised_3d_gauss_cartesian(Ds)

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = gauss_full * luminosity_term

        # self.full = full_expression

        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)

        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            self.H_0_pdf = self.H_0_pdf/P_det_total_power
        
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf
        

    def H_0_inference_2d_perfect_survey_vectorised_gaussian_radius(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        gaussian_radius_term = self.get_vectorised_gaussian_rads(Ds)

        vmf = self.get_vectorised_vm()

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = gaussian_radius_term * vmf * luminosity_term
        
        H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf_single_event = np.reciprocal((np.sum(H_0_pdf_single_event, axis=1))*self.H_0_increment)[:, np.newaxis] * H_0_pdf_single_event
        self.H_0_pdf = np.sum(np.log(H_0_pdf_single_event), axis=0)
        
        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power_log = self.SurveyAndEventData.detected_event_count * np.log(P_det_total)
            self.H_0_pdf = self.H_0_pdf - P_det_total_power_log
        
        self.H_0_pdf -= np.max(self.H_0_pdf)
        self.H_0_pdf = np.exp(self.H_0_pdf)
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey_vectorised_gaussian_radius(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        gaussian_radius_term = self.get_vectorised_gaussian_rads(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = self.lum_term[self.event_distribution_inf](Ds)

        full_expression = gaussian_radius_term * vmf * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf_single_event = np.reciprocal((np.sum(H_0_pdf_single_event, axis=1))*self.H_0_increment)[:, np.newaxis] * H_0_pdf_single_event
        self.H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        
        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power_log = self.SurveyAndEventData.detected_event_count * np.log(P_det_total)
            self.H_0_pdf = self.H_0_pdf - P_det_total_power_log
        

        self.H_0_pdf -= np.max(self.H_0_pdf)
        self.H_0_pdf = np.exp(self.H_0_pdf)
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        return self.H_0_pdf

    def get_mean(self):
        if self.distribution_calculated:
            self.mean = np.sum(self.H_0_pdf*self.H_0_range)*self.H_0_increment
            return self.mean
        else:
            self.H_0_pdf()
            self.get_mean()

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
                    D = (self.SurveyAndEventData.detected_redshifts[g]) * self.c/ H_0
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
                    D = (self.SurveyAndEventData.detected_redshifts[g]) * self.c/ H_0
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

    def H_0_inference_gamma(self):
        log_gamma_marginalised = np.zeros(len(self.H_0_pdf))
        expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1(H_0)
            expected_event_num_divded_by_gamma[H_0_index] = N1
            scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
            scaled_gamma_lower_lim = self.SurveyAndEventData.gamma_lower_lim*N1
            Nhat = self.SurveyAndEventData.detected_event_count
            log_gamma_marginalised[H_0_index] = np.log(gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim))
        log_gamma_marginalised -= np.max(log_gamma_marginalised)
        self.gamma_marginalised = np.exp(log_gamma_marginalised)
        self.expected_event_num_divded_by_gamma = expected_event_num_divded_by_gamma
        return self.gamma_marginalised

    def H_0_inference_2d_gamma_gaussian(self):
        log_gamma_marginalised = np.zeros(len(self.H_0_pdf))
        expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1_gaussian(H_0)
            expected_event_num_divded_by_gamma[H_0_index] = N1
            scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
            scaled_gamma_lower_lim = self.SurveyAndEventData.gamma_lower_lim*N1
            Nhat = self.SurveyAndEventData.detected_event_count
            log_gamma_marginalised[H_0_index] = np.log(gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim))
        log_gamma_marginalised -= np.max(log_gamma_marginalised)
        self.gamma_marginalised = np.exp(log_gamma_marginalised)
        self.expected_event_num_divded_by_gamma = expected_event_num_divded_by_gamma
        return self.gamma_marginalised

    def H_0_inference_gamma_gaussian(self):
        log_gamma_marginalised = np.zeros(len(self.H_0_pdf))
        expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1_gaussian(H_0)
            expected_event_num_divded_by_gamma[H_0_index] = N1
            scaled_gamma_upper_lim = self.SurveyAndEventData.gamma_upper_lim*N1
            scaled_gamma_lower_lim =  self.SurveyAndEventData.gamma_lower_lim*N1
            Nhat = self.SurveyAndEventData.detected_event_count
            log_gamma_marginalised[H_0_index] = np.log(gammainc(Nhat, scaled_gamma_upper_lim)-gammainc(Nhat, scaled_gamma_lower_lim))
        log_gamma_marginalised -= np.max(log_gamma_marginalised)
        self.gamma_marginalised = np.exp(log_gamma_marginalised)
        self.expected_event_num_divded_by_gamma = expected_event_num_divded_by_gamma
        return self.gamma_marginalised
    
    def H_0_inference_gamma_known(self):
        gamma_marginalised = np.zeros(len(self.H_0_pdf))
        expected_event_num_divded_by_gamma = np.zeros(len(self.H_0_pdf))
        for H_0_index, H_0 in enumerate(self.H_0_range):
            N1 = self.calc_N1(H_0)
            expected_event_num_divded_by_gamma[H_0_index] = N1
            Nhat = self.SurveyAndEventData.detected_event_count
            gamma_marginalised[H_0_index] = ss.poisson.pmf(k = Nhat, mu = N1*self.SurveyAndEventData.event_rate)
        self.gamma_marginalised = gamma_marginalised
        self.expected_event_num = expected_event_num_divded_by_gamma*self.SurveyAndEventData.event_rate
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
        cdf = 1-np.power(1 + np.power((max_D/lam),c),-k)
        return cdf

    def burr_cdf_x(self, x, lam):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        cdf = 1-(1 + (x/lam)**c)**-k
        return cdf

    def get_vectorised_burr(self, Ds):
        Ds_tile = np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1))

        recip_Ds_tile = np.reciprocal(Ds_tile)
        u_r = np.sqrt(np.einsum('ij,ij->i', self.SurveyAndEventData.BH_detected_coords,
                                self.SurveyAndEventData.BH_detected_coords))[:, np.newaxis, np.newaxis]

        # u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))[:, np.newaxis, np.newaxis]

        omegas = recip_Ds_tile * u_r

        burr_term1 = np.power(omegas, self.SurveyAndEventData.BVM_c - 1)
        burr_term2 = np.power(1 + np.power(omegas, self.SurveyAndEventData.BVM_c), - self.SurveyAndEventData.BVM_k - 1)

        burr_full = self.SurveyAndEventData.BVM_k * self.SurveyAndEventData.BVM_c * recip_Ds_tile * burr_term1 * burr_term2
        return burr_full

    def get_vectorised_gaussian_rads(self, Ds):

        Ds_tile = np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1))

        u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))[:, np.newaxis, np.newaxis]

        gaussian_coefficient = 1/(np.sqrt(2*np.pi) * self.SurveyAndEventData.noise_sigma)

        gaussian_full = gaussian_coefficient * np.exp(-0.5*np.square((u_r - Ds_tile) / self.SurveyAndEventData.noise_sigma))

        return gaussian_full
    
    def get_vectorised_vm(self):
        kappa = self.SurveyAndEventData.BVM_kappa
        vm_C = kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))

        u_x = self.SurveyAndEventData.BH_detected_coords[:, 0]
        u_y = self.SurveyAndEventData.BH_detected_coords[:, 1]

        u_phi = np.arctan2(u_y, u_x)[:, np.newaxis]

        X = self.SurveyAndEventData.detected_coords[:, 0]
        Y = self.SurveyAndEventData.detected_coords[:, 1]

        phi = np.tile(np.arctan2(Y, X), (self.SurveyAndEventData.detected_event_count, 1))

        cos_phi_diff = np.cos(phi - u_phi)

        vm = vm_C * np.exp(kappa * cos_phi_diff)[:, np.newaxis,:]
        return vm    
    
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

    def get_vectorised_normal_burr_int(self, us, ss):
        print('Reached Burr Normal integral')
        U = np.tile(us[np.newaxis, :], (self.SurveyAndEventData.detected_event_count, 1, 1))
        S = np.tile(ss[np.newaxis, :], (self.SurveyAndEventData.detected_event_count, 1, 1))
        d = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))[:, np.newaxis, np.newaxis]
        D = np.tile(d, (1, len(self.H_0_range), len(self.SurveyAndEventData.detected_redshifts)))

        integral = self.vectorised_integrate_on_grid(self.P_D_gal, 0, np.inf)
        burr_full = integral(D, U, S)
        print('Ended Burr Normal integral')
        return burr_full
    
    def get_vectorised_3d_gauss_cartesian(self, Ds):
        Ds_tile = np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1))

        #recip_Ds_tile = np.reciprocal(Ds_tile)

        u_x = self.SurveyAndEventData.BH_detected_coords[:, 0]
        u_y = self.SurveyAndEventData.BH_detected_coords[:, 1]
        u_z = self.SurveyAndEventData.BH_detected_coords[:, 2]

        uu_x = u_x[:, np.newaxis, np.newaxis]
        uu_y = u_y[:, np.newaxis, np.newaxis]
        uu_z = u_z[:, np.newaxis, np.newaxis]
        
        #uu_x = np.tile(u_x, (self.SurveyAndEventData.detected_event_count, 1, 1))
        #uu_y = np.tile(u_y, (self.SurveyAndEventData.detected_event_count, 1, 1))
        #uu_z = np.tile(u_z, (self.SurveyAndEventData.detected_event_count, 1, 1))


        X = self.SurveyAndEventData.detected_coords[:, 0]
        Y = self.SurveyAndEventData.detected_coords[:, 1]
        Z = self.SurveyAndEventData.detected_coords[:, 2]
        XY = np.sqrt((X) ** 2 + (Y) ** 2)

        phi = np.tile(np.arctan2(Y, X), (self.SurveyAndEventData.detected_event_count, 1))
        theta = np.tile(np.arctan2(XY, Z), (self.SurveyAndEventData.detected_event_count, 1))

        #x_g = Ds_tile * np.sin(theta)[:,np.newaxis,:] * np.cos(phi)[np.newaxis,:,:]
        #y_g = Ds_tile * np.sin(theta)[:,np.newaxis,:] * np.sin(phi)[np.newaxis,:,:]
        #z_g = Ds_tile * np.cos(theta)[:,np.newaxis,:]

        sin_cos = np.sin(theta) * np.cos(phi)
        sin_sin = np.sin(theta) * np.sin(phi)
        cos = np.cos(theta)
        x_g = Ds_tile * sin_cos[:,np.newaxis,:]
        y_g = Ds_tile * sin_sin[:,np.newaxis,:]
        z_g = Ds_tile * cos[:,np.newaxis,:]
        #x_g = Ds_tile * np.sin(theta)[np.newaxis,np.newaxis,:] * np.cos(phi)[np.newaxis,np.newaxis,:]
        #y_g = Ds_tile * np.sin(theta)[np.newaxis,np.newaxis,:] * np.sin(phi)[np.newaxis,np.newaxis,:]
        #z_g = Ds_tile * np.cos(theta)[np.newaxis,np.newaxis,:]

        omega_x = uu_x - x_g
        omega_y = uu_y - y_g
        omega_z = uu_z - z_g

        gauss_term = np.square(omega_x) + np.square(omega_y) + np.square(omega_z)

        gauss_full = np.exp(-gauss_term/(2*self.SurveyAndEventData.noise_sigma**2))

        return gauss_full

    def get_vectorised_3d_gauss_cartesian_version_2(self, Ds):
        # gives the same result but with slightly different code

        u_x = self.SurveyAndEventData.BH_detected_coords[:, 0]
        u_y = self.SurveyAndEventData.BH_detected_coords[:, 1]
        u_z = self.SurveyAndEventData.BH_detected_coords[:, 2]

        uu_x = u_x[:, np.newaxis, np.newaxis]
        uu_y = u_y[:, np.newaxis, np.newaxis]
        uu_z = u_z[:, np.newaxis, np.newaxis]
        
        #uu_x = np.tile(u_x, (self.SurveyAndEventData.detected_event_count, 1, 1))
        #uu_y = np.tile(u_y, (self.SurveyAndEventData.detected_event_count, 1, 1))
        #uu_z = np.tile(u_z, (self.SurveyAndEventData.detected_event_count, 1, 1))


        X = self.SurveyAndEventData.detected_coords[:, 0]
        Y = self.SurveyAndEventData.detected_coords[:, 1]
        Z = self.SurveyAndEventData.detected_coords[:, 2]
        XY = np.sqrt((X) ** 2 + (Y) ** 2)

        phi = np.arctan2(Y, X)
        theta = np.arctan2(XY, Z)

        sin_cos = np.sin(theta) * np.cos(phi)
        sin_sin = np.sin(theta) * np.sin(phi)
        cos = np.cos(theta)
        x_g = Ds * sin_cos[np.newaxis,:]
        y_g = Ds * sin_sin[np.newaxis,:]
        z_g = Ds * cos[np.newaxis,:]

        omega_x = uu_x - x_g[np.newaxis,:]
        omega_y = uu_y - y_g[np.newaxis,:]
        omega_z = uu_z - z_g[np.newaxis,:]

        gauss_term = np.square(omega_x) + np.square(omega_y) + np.square(omega_z)

        gauss_full = np.exp(-gauss_term/(2*self.SurveyAndEventData.noise_sigma**2))

        return gauss_full



    def calc_N1(self, H_0):
        N1 = 0
        for g_i, flux in enumerate(self.SurveyAndEventData.fluxes):
            D_gi = self.SurveyAndEventData.detected_redshifts[g_i] * self.c / H_0
            if self.SurveyAndEventData.dimension == 2:
                luminosity = 2*np.pi*flux*D_gi
            else:
                luminosity = 4*np.pi*flux*D_gi**2
            N1 += luminosity*self.burr_cdf(lam = D_gi)
        N1 *= self.SurveyAndEventData.sample_time
        return N1
    
    def calc_N1_gaussian(self, H_0):
        N1 = 0
        for g_i, flux in enumerate(self.SurveyAndEventData.fluxes):
            D_gi = self.SurveyAndEventData.detected_redshifts[g_i]/H_0
            if self.SurveyAndEventData.dimension == 2:
                luminosity = 2*np.pi*flux*D_gi
            else:
                luminosity = 4*np.pi*flux*D_gi**2
            N1 += luminosity*ss.norm.cdf(x = self.SurveyAndEventData.max_D ,loc = D_gi, scale = self.SurveyAndEventData.noise_sigma)
        N1 *= self.SurveyAndEventData.sample_time
        return N1
    
    
    def P_D_gal(self,x,D,U,S):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        return np.exp(-(x-U)**2/(2*S**2))/((x**(c))*(1+(D/x)**c)**(k+1))

    def vectorised_integrate_on_grid(self, func, lo, hi):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n,m,l: quad(func, lo, hi, (n,m,l))[0])


    ### Completeness

    def BB1_p(self,x,b,u,l):
        b = b-2
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
        p =(C/u)*(1-np.exp(-x/l))*((x/u)**b)*np.exp(-x/u)
        return p

    def var_lum_prop(self, L):
        return L
    
    def var_lum_const(self, L):
        return 1
    
    def var_schechter(self, L):
        return sps.gamma.pdf(L , 0.3, scale=1)
    
    def var_BB1(self, L):
        return self.BB1_p(L, -1.5, 1, 0.1)
    
    def p_G_integrad(self, L, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * self.lum_term_integrand[self.event_distribution_inf](L) * self.lum_function_integrand[self.lum_function_inf](L)

    def p_G(self, H_0_values):
        # Need to integrate one for every H_0 value
        p_g = self.vectorised_integrate_2d(self.p_G_integrad, 0, np.inf, lambda z: 4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z/H_0_values)**2, np.inf)(H_0_values) / (
                self.vectorised_integrate_2d(self.p_G_integrad, 0, np.inf, 0, np.inf)(H_0_values))
        return p_g
    
    def vectorised_integrate_2d(self, func, lo, hi, gfun, hfun):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n: dblquad(func, lo, hi, gfun, hfun, [n])[0])


    def p_G_trial(self, H_0_values):
        # Need to integrate one for every H_0 value
        p_g = np.zeros(len(H_0_values))
        for i in range(len(H_0_values)):
            p_g[i] = dblquad(self.p_G_integrad, 0, np.inf, lambda z: 4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z/H_0_values[i])**2, np.inf, args=[H_0_values[i],], epsabs=0.00001)[0] / (
                    dblquad(self.p_G_integrad, 0, np.inf, 0, np.inf, args=[H_0_values[i],],  epsabs=0.00001)[0])
        return p_g

    ### Better completeness

    def denominator_integrand(self, z , H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0)

    def schecher_lum_uni(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammaincc(self.SurveyAndEventData.alpha, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)

    def schecher_lum_prop(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammaincc(self.SurveyAndEventData.alpha + 1, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)

    def BB1_lum_uni(self, z, H_0):
        return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star) - ((4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/self.SurveyAndEventData.L_star)**(1 + self.SurveyAndEventData.beta)) * np.exp(-4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/self.SurveyAndEventData.L_star)/gamma_function(self.SurveyAndEventData.beta + 2) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta)) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) - ((4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum))**(1 + self.SurveyAndEventData.beta)) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) / gamma_function(2 + self.SurveyAndEventData.beta) ))

    def BB1_lum_prop(self, z, H_0):
        return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta)) * gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) )
    
    def p_x_gw_G_bar(self, z, H_0, x_gw):
        return 

    def P_G_single(self, H_0_values):
        iss = np.zeros(len(H_0_values))
        for i in range(len(H_0_values)):
            iss[i] = quad(self.BB1_lum_uni, 0, np.inf, args=(H_0_values[i],))[0] / quad(self.denominator_integrand, 0, np.inf, args=(H_0_values[i],))[0]
        return iss





#%%




# from Components.EventGenerator import EventGenerator
# # # '''
# Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.01, event_rate=200,
#                         luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
#                         cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=100,
#                         event_distribution="Proportional", noise_distribution = "BVMF_eff", contour_type = "BVM", redshift_noise_sigma = 0.0,
#                         resolution=100, plot_contours = False, seed = 1)
# print(Gen.detected_event_count)
# print(len(Gen.detected_luminosities))
# print(Gen.total_event_count)
#
# for i in range(10):
#     print(Gen.poisson_event_count())
#
# # Gen.plot_universe_and_events()
# Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data, survey_type='perfect', resolution_H_0=500)
# #
# # CDF = []
# # X = np.arange(1,100)
# # for elem in X:
# #     CDF.append(Y.burr_cdf(elem))
# #
# # plt.plot(X, CDF)
# # plt.yscale("log")
# # Y2 = Inference(Data, survey_type='perfect')
# plt.plot(Y.H_0_range, Y.H_0_Prob())
# plt.plot(Y.H_0_range, Y2.H_0_Prob())
# plt.show()

# print("efficient done")
#
# '''




# %%
# 
# import numpy as np    
# from scipy.integrate import quad
# 
# def f(x,u,s):
#     return np.exp(-(x-u)**2/(2*s**2))
# 
# def g(x,D,c,k):
#     return 1/((x**c)*(1+(D/x)**c)**(k+1))
# 
# def h(x,D,u,s):
#     c = 15
#     k = 2
#     return f(x,u,s)*g(x,D,c,k)
# 
# c = 15
# k = 2
# s = 0.05
# u = np.arange(1,1000)
# D = np.arange(1,1000,20)
# 
# #for i in D:
# #    for j in u:
# #        r = quad(h,0,np.inf, args=(i,j,s,c,k))[0]
# #        print(r)
# 
# #%%
# 
# def bgauss(D,u,s):
#     c = 15
#     k = 2
#     return quad(h,0,np.inf,args=(D,u,s))[0]
# 
# vec_pos = np.vectorize(bgauss)
# 
# 
# #%%
# 
# def integrate_on_grid(func, lo, hi):
#     """Returns a callable that can be evaluated on a grid."""
#     return np.vectorize(lambda n,m,l: quad(func, lo, hi, (n,m,l))[0])
# 
# #Ds, us = np.mgrid[1:1000:20, 1:1000:1]
# c = 15
# k = 2
# #I = integrate_on_grid(h, 0, np.inf, c,k)(Ds,us)

# %%
