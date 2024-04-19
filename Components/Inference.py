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
from scipy.special import gammaincc, gammainc
from scipy.special import gamma as gamma_function
from mpmath import gammainc as mp_gammainc
import scipy.stats as ss
import time



class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData, gamma = True, vectorised = True, event_distribution_inf='Proportional', lum_function_inf = 'Full-Schechter', hubble_law_inf='linear', gauss=False, p_det=True, poster=False,
                 survey_type ='perfect', resolution_H_0=100, H_0_Min = 50, H_0_Max = 100, resolution_q_0=50, q_0_Min = -1.99, q_0_Max = 0.99, gamma_known = False, gauss_type = "Cartesian", flux_threshold = 0):

        self.SurveyAndEventData = SurveyAndEventData
        self.c = self.SurveyAndEventData.c
        self.flux_threshold = flux_threshold
        if self.flux_threshold !=0:
            self.completeness_method = "incomplete"
        else: self.completeness_method = ""

        self.distribution_calculated = False
        self.H_0_Min = H_0_Min
        self.H_0_Max = H_0_Max

        self.q_0_Min = q_0_Min
        self.q_0_Max = q_0_Max

        self.survey_type = survey_type
        self.event_distribution_inf = event_distribution_inf
        self.lum_function_inf = lum_function_inf
        self.hubble_law_inf = hubble_law_inf

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

        self.resolution_q_0 = resolution_q_0
        self.q_0_pdf = np.zeros(self.resolution_q_0)
        self.q_0_range = np.linspace(self.q_0_Min, self.q_0_Max, self.resolution_q_0)
        self.q_0_increment = self.q_0_range[1] - self.q_0_range[0]

        self.H_0_pdf_single_event = np.zeros(shape = (len(self.SurveyAndEventData.BH_detected_coords), self.resolution_H_0))
        self.event_selection = self.SurveyAndEventData.event_distribution

        self.inference_method = dict({"perfect2d":self.H_0_inference_2d_perfect_survey,
                                # "imperfect2d": self.H_0_inference_2d_imperfect_survey,
                                # "perfect3d": self.H_0_inference_3d_perfect_survey,
                                # "imperfect3d": self.H_0_inference_3d_imperfect_survey,
                                "perfectvectorised2d": self.H_0_inference_2d_perfect_survey_vectorised,
                                "perfectvectorised3d": self.H_0_inference_3d_perfect_survey_vectorised,
                                "perfectvectorised3dquadratic": self.H_0_inference_3d_perfect_survey_vectorised_quadratic,
                                "perfectvectorised3dincomplete": self.H_0_inference_3d_perfect_survey_vectorised_incomplete,
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

        self.lum_term_uncertain_z = dict({'Random':self.get_lum_term_random_x, 
                                    'Proportional': self.get_lum_term_proportional_x})

        self.flux_contribution = dict({'Random':self.flux_if_random, 
                                    'Proportional': self.flux_if_prop})

        #self.lum_term_integrand = dict({'Random':self.var_lum_const, 
        #                      'Proportional': self.var_lum_prop})

        #self.lum_function_integrand = dict({'Full-Schechter':self.var_BB1,
        #                                    'Shoddy-Schechter': self.var_schechter})

        self.comp_integrand_name = self.lum_function_inf + '_' + self.event_distribution_inf

        self.upper_integrand_p_G = dict({'Shoddy-Schechter_Random': self.schecher_lum_uni,
                                         'Shoddy-Schechter_Proportional': self.schecher_lum_prop,
                                         'Full-Schechter_Random': self.BB1_lum_uni,
                                         'Full-Schechter_Proportional': self.BB1_lum_prop})

        self.numerator_integrand_p_x_GW = dict({'Shoddy-Schechter_Random': self.gw_numerator_schecher_lum_uni,
                                         'Shoddy-Schechter_Proportional': self.gw_numerator_schecher_lum_prop,
                                         'Full-Schechter_Random': self.gw_numerator_BB1_lum_uni,
                                         'Full-Schechter_Proportional': self.gw_numerator_BB1_lum_prop})

        self.denominator_integrand_p_x_GW = dict({'Shoddy-Schechter_Random': self.gw_denominator_schecher_lum_uni,
                                         'Shoddy-Schechter_Proportional': self.gw_denominator_schecher_lum_prop,
                                         'Full-Schechter_Random': self.gw_denominator_BB1_lum_uni,
                                         'Full-Schechter_Proportional': self.gw_denominator_BB1_lum_prop})

        #if self.SurveyAndEventData.dimension==3 and self.survey_type == "perfect" and self.vectorised:
        #    self.inference_method_name = self.survey_type + "vectorised" + str(self.SurveyAndEventData.dimension) + "d"

        if not self.gauss:
            self.inference_method_name = self.survey_type + "vectorised" + str(self.SurveyAndEventData.dimension) + "d"+self.completeness_method
        elif self.gauss:
            self.inference_method_name = self.survey_type + "vectorised" + str(
                self.SurveyAndEventData.dimension) + "d" + "Gauss" + self.gauss_type

        if self.poster:
            self.inference_method_name+='poster'
        
        if self.hubble_law_inf == 'quadratic':
            self.inference_method_name+='quadratic'

        self.g_H_0 = dict()

        self.countour = dict({"gauss": self.gauss_p_hat_g_true,
                              "BVM": self.BVM_p_hat_g_true})


    def H_0_Prob(self):
        self.distribution_calculated = True
        if self.SurveyAndEventData.detected_event_count == 0:
            self.H_0_pdf = np.ones(self.resolution_H_0)
            self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
            return self.H_0_pdf
        else:
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

    def get_lum_term_proportional(self, redshifts, initial=0, final=0):
        if self.hubble_law_inf == 'linear':
            term = np.square(redshifts) * self.SurveyAndEventData.fluxes
        elif self.hubble_law_inf == 'quadratic':
            q_0 = np.tile(self.q_0_range[initial:final], (self.resolution_H_0,1))
            term = np.square((redshifts) * (1 + 0.5*redshifts*(1-q_0[:,:,np.newaxis]))) * self.SurveyAndEventData.fluxes
        return term

    def get_lum_term_random(self, Ds, initial=0, final=0):
        return 1
    
    def get_lum_term_proportional_x(self, x):
        return x**2
    
    def get_lum_term_random_x(self, x):
        return 1
    
    def flux_if_prop(self, F):
        return F
    
    def flux_if_random(self, F):
        return 1

    def H_0_inference_2d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        burr_full = self.get_vectorised_burr(Ds)

        vm = self.get_vectorised_vm()

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

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

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

        full_expression = burr_full * vmf * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        #self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)
        '''
        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            self.H_0_pdf = self.H_0_pdf/P_det_total_power
        '''
        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            self.H_0_pdf_single_event = self.H_0_pdf_single_event / P_det_total

        # f = np.vectorize(math.frexp)
        # split = f(self.H_0_pdf_single_event)
        # flo = split[0]
        # ex = split[1]
        # p_flo = np.prod(flo, axis=0)
        # p_ex = np.sum(ex, axis=0)
        # scaled_ex = p_ex - np.max(p_ex)
        # scaled_flo = p_ex / p_flo[np.argmax(p_ex)]
        # self.H_0_pdf = scaled_flo * (0.5 ** (-1 * scaled_ex))

        self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        self.H_0_pdf = np.exp(self.log_H_0_pdf - np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)]))

        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        self.H_0_pdf_single_event /= np.sum(self.H_0_pdf_single_event, axis=1)[:, np.newaxis] * (self.H_0_increment)

        return self.H_0_pdf


    def H_0_inference_3d_perfect_survey_vectorised_quadratic(self):
        h_0_recip = np.reciprocal(self.H_0_range)
        H_0_recip = np.tile(h_0_recip, (self.resolution_q_0,1)).T
        q_0 = np.tile(self.q_0_range, (self.resolution_H_0,1))

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, self.resolution_q_0, 1))

        Ds = redshifts * self.c * ( 1 + 0.5 * redshifts * (1-q_0[:,:,np.newaxis] )) * H_0_recip[:,:,np.newaxis]

        burr_full = self.get_vectorised_burr_quadratic(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

        full_expression = burr_full * vmf[:,:,np.newaxis,:] * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=3)
        #self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)
        '''
        if self.p_det:
            p_det_vec = luminosity_term * self.get_p_det_vec(Ds)
            P_det_total = np.sum(p_det_vec, axis=1)
            self.P_det_total = P_det_total
            P_det_total_power = np.power(P_det_total, self.SurveyAndEventData.detected_event_count)
            self.H_0_pdf = self.H_0_pdf/P_det_total_power
        '''
        if self.p_det:
            p_det_vec = self.get_p_det_vec(Ds) * luminosity_term
            P_det_total = np.sum(p_det_vec, axis=2)
            self.P_det_total = P_det_total
            self.H_0_pdf_single_event = self.H_0_pdf_single_event / P_det_total

        f = np.vectorize(math.frexp)
        split = f(self.H_0_pdf_single_event)
        flo = split[0]
        ex = split[1]
        p_flo = np.prod(flo, axis=0)
        p_ex = np.sum(ex, axis=0)
        scaled_ex = p_ex - np.max(p_ex)
        scaled_flo = p_ex / p_flo.flatten()[np.argmax(p_ex)]
        self.H_0_pdf = scaled_flo * (0.5 ** (-1 * scaled_ex))

        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment) * (self.q_0_increment)

        #self.H_0_pdf_single_event /= np.sum(self.H_0_pdf_single_event, axis=1)[:, np.newaxis] * (self.H_0_increment)

        return self.H_0_pdf


    def poster_H_0_inference_3d_perfect_survey_vectorised(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = redshifts * H_0_recip * self.c

        burr_full = self.get_vectorised_burr(Ds)

        vmf = self.get_vectorised_vmf()

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

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

        self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        self.H_0_pdf = np.exp(self.log_H_0_pdf - np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)]))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        
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

        #luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

        flux_term = self.flux_contribution[self.event_distribution_inf](self.SurveyAndEventData.fluxes)
        weight = flux_term / (self.SurveyAndEventData.redshift_noise_sigma**2 + self.SurveyAndEventData.detected_redshifts**2)
        #full_expression = burr_full * vmf * luminosity_term
        
        full_expression = burr_full * vmf * weight

        # self.full = full_expression

        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)

        # self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        # self.H_0_pdf = np.exp(self.log_H_0_pdf - np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)]))
        # self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        if self.p_det:
            P_det_full = self.get_normal_P_det_int(us,ss)
            p_det_vec = P_det_full * weight
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

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

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

        self.log_H_0_pdf = np.sum(np.log(self.H_0_pdf_single_event), axis=0)
        self.H_0_pdf = np.exp(self.log_H_0_pdf - np.min(self.log_H_0_pdf[np.isfinite(self.log_H_0_pdf)]))
        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)
        return self.H_0_pdf
        

    def H_0_inference_2d_perfect_survey_vectorised_gaussian_radius(self):
        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        gaussian_radius_term = self.get_vectorised_gaussian_rads(Ds)

        vmf = self.get_vectorised_vm()

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

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

        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

        full_expression = gaussian_radius_term * vmf * luminosity_term
        
        self.H_0_pdf_single_event = np.sum(full_expression, axis=2)
        self.H_0_pdf_single_event = np.reciprocal((np.sum(self.H_0_pdf_single_event, axis=1))*self.H_0_increment)[:, np.newaxis] * self.H_0_pdf_single_event
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
        #Technically we should have a gaussian cdf method here (and gaussian with jacobean cdf)
        # - it has negligible effect so haven't done it.
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
    
    def burr_pdf(self, x, lam):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        pdf = (c*k/lam) * (x/lam)**(c-1) * (1 + (x/lam)**c)**(-k-1)
        return pdf

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
    
    def get_vectorised_burr_quadratic(self, Ds):
        #Ds_tile = np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1, 1))

        #recip_Ds_tile = np.reciprocal(Ds_tile)
        recip_Ds_tile = np.reciprocal(np.tile(Ds, (self.SurveyAndEventData.detected_event_count, 1, 1, 1)))
        u_r = np.sqrt(np.einsum('ij,ij->i', self.SurveyAndEventData.BH_detected_coords,
                                self.SurveyAndEventData.BH_detected_coords))[:, np.newaxis, np.newaxis, np.newaxis]

        # u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))[:, np.newaxis, np.newaxis]

        #omegas = recip_Ds_tile * u_r

        #burr_term1 = np.power(omegas, self.SurveyAndEventData.BVM_c - 1)
        #burr_term2 = np.power(1 + np.power(omegas, self.SurveyAndEventData.BVM_c), - self.SurveyAndEventData.BVM_k - 1)

        #burr_full = self.SurveyAndEventData.BVM_k * self.SurveyAndEventData.BVM_c * recip_Ds_tile * burr_term1 * burr_term2
        
        burr_full = self.SurveyAndEventData.BVM_k * self.SurveyAndEventData.BVM_c * recip_Ds_tile * np.power(recip_Ds_tile * u_r, self.SurveyAndEventData.BVM_c - 1) * np.power(1 + np.power(recip_Ds_tile * u_r, self.SurveyAndEventData.BVM_c), - self.SurveyAndEventData.BVM_k - 1)
        
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

        integral = self.vectorised_integrate_on_grid_eps(self.L_GW_z_uncertain, 0.0001, 1500, 0.0001)#, points=(self.SurveyAndEventData.max_D,))
        burr_full = integral(D, U, S)
        print('Ended Burr Normal integral')
        return burr_full
    
    def get_normal_P_det_int(self, us, ss):
        print('Reached P_det Normal integral')
        #U = np.tile(us[np.newaxis, :], (self.SurveyAndEventData.detected_event_count, 1, 1))
        #S = np.tile(ss[np.newaxis, :], (self.SurveyAndEventData.detected_event_count, 1, 1))

        integral = self.vectorised_integrate_on_grid_2_arg_eps(self.P_det_z_uncertain, 0.0001, 1500, 0.0001)#, points=(self.SurveyAndEventData.max_D,))
        p_det_full = integral(us, ss)
        print('Ended P_det Normal integral')
        return p_det_full

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

    #def L_GW_z_uncertain(self,x,D,U,S):
    #    c = self.SurveyAndEventData.BVM_c
    #    k = self.SurveyAndEventData.BVM_k
    #    return self.lum_term_uncertain_z[self.event_distribution_inf](x) * np.exp(-(x-U)**2/(2*S**2)) * (x**(3-c)) * (1 + (D/x)**c)**(-1-k)

    def L_GW_z_uncertain(self,x,D,U,S):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        return self.lum_term_uncertain_z[self.event_distribution_inf](x) * np.exp(-(x-U)**2/(2*S**2)) *x**2 * ((D/x)**(c-1)) * (1 + (D/x)**c)**(-1-k)


    def P_det_z_uncertain(self,x,U,S):
        c = self.SurveyAndEventData.BVM_c
        k = self.SurveyAndEventData.BVM_k
        D_max = self.SurveyAndEventData.max_D
        return self.lum_term_uncertain_z[self.event_distribution_inf](x) * np.exp(-(x-U)**2/(2*S**2)) * x**2 *(1 - (1 + (D_max/x)**c)**(-k))


    def vectorised_integrate_on_grid(self, func, lo, hi):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n,m,l: quad(func, lo, hi, (n,m,l))[0])

    def vectorised_integrate_on_grid_eps(self, func, lo, hi, epsrel):#, points):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n,m,l: quad(func, lo, hi, (n,m,l), epsrel=epsrel, epsabs=0)[0])

    def vectorised_integrate_on_grid_1_arg(self, func, lo, hi):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n: quad(func, lo, hi, (n,))[0])

    def vectorised_integrate_on_grid_1_arg_eps(self, func, lo, hi, epsrel):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n: quad(func, lo, hi, (n,), epsrel=epsrel, epsabs=0)[0])

    def vectorised_integrate_on_grid_2_arg(self, func, lo, hi):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n,l: quad(func, lo, hi, (n,l))[0])
    
    def vectorised_integrate_on_grid_2_arg_eps(self, func, lo, hi, epsrel):#, points):
        """Returns a callable that can be evaluated on a grid."""
        return np.vectorize(lambda n,l: quad(func, lo, hi, (n,l), epsrel=epsrel, epsabs=0)[0])

    ### Completeness

    def H_0_inference_3d_perfect_survey_vectorised_incomplete(self):

        H_0_recip = np.reciprocal(self.H_0_range)[:, np.newaxis]

        redshifts = np.tile(self.SurveyAndEventData.detected_redshifts, (self.resolution_H_0, 1))

        Ds = self.c * redshifts * H_0_recip

        # In Survey Terms (above Fth)

        upper_integrand_func_G = self.upper_integrand_p_G[self.comp_integrand_name]
        integral_1 = self.vectorised_integrate_on_grid_1_arg(upper_integrand_func_G, 0, 10)
        integral_2 = self.vectorised_integrate_on_grid_1_arg(self.denominator_integrand, 0, 10)   
        P_G = integral_1(self.H_0_range)/integral_2(self.H_0_range)
        # or P_G = integral_1(70)/integral_2(70)
        # or interpolate

        #Initiate the Cube

        burr_full = self.get_vectorised_burr(Ds)
        vmf = self.get_vectorised_vmf()
        luminosity_term = self.lum_term[self.event_distribution_inf](redshifts)

        #Collapse the Cube
        P_GWdata_given_G = np.sum(burr_full * vmf * luminosity_term, axis=2) 
        #P_GWdata_given_G = P_GWdata_given_G * P_G

        if self.p_det:
            p_det_vec_given_G = luminosity_term * self.get_p_det_vec(Ds)  # Change the final term
            P_det_total_given_G = np.sum(p_det_vec_given_G, axis=1)
            # Note it is the same for all galaxies - collapse the cube first
            P_GWdata_given_G = np.divide(P_GWdata_given_G, P_det_total_given_G)

        self.P_GWdata_given_G_before = P_GWdata_given_G
        
        P_GWdata_given_G = P_GWdata_given_G * P_G

        # Out of Survey Terms (below Fth)
    
        P_Gbar = 1 - P_G

        # This should be dealt with after the galaxy side of the cube is collapsed
        u_r = np.sqrt(np.sum(np.square(self.SurveyAndEventData.BH_detected_coords), axis=1))
        U_R_grid = np.tile(u_r[:,np.newaxis], (1,self.resolution_H_0))
        H_0_grid = np.tile(self.H_0_range[np.newaxis,:], (self.SurveyAndEventData.detected_event_count,1))

        gw_denominator_func = self.denominator_integrand_p_x_GW[self.comp_integrand_name]
        gw_numerator_func = self.numerator_integrand_p_x_GW[self.comp_integrand_name]
        
        integral_3 = self.vectorised_integrate_on_grid_2_arg_eps(gw_numerator_func, 0, 10, 1/10**7)
        integral_4 = self.vectorised_integrate_on_grid_1_arg_eps(gw_denominator_func, 0, 10, 1/10**7)

        P_GWdata_given_Gbar = integral_3(U_R_grid, H_0_grid) / integral_4(self.H_0_range)  # Replace this term

        # if not really dependent on H_0 this should be
        # P_GWdata_given_Gbar = integral_3(u_r, 70) / integral_4(70) 
        self.P_GWdata_given_Gbar_before = P_GWdata_given_Gbar
        
        P_GWdata_given_Gbar = P_GWdata_given_Gbar * P_Gbar

        # for now it's easier if this isn't an option since the integration expressions have terms i did not incluse since they cancelled with p_det
        '''
        if self.p_det:
            p_det_vec_given_Gbar = luminosity_term * self.get_p_det_vec(Ds)  # Change the final term
            P_det_total_given_Gbar = np.sum(p_det_vec_given_Gbar, axis=1)
            # Note it is the same for all galaxies - collapse the cube first
            P_GWdata_given_Gbar = np.divide(P_GWdata_given_Gbar, P_det_total_given_Gbar)
        '''

        # likelihood_expression = P_GWdata_given_G + P_GWdata_given_Gbar
        
        self.P_G = P_G
        self.P_GWdata_given_G = P_GWdata_given_G
        self.P_GWdata_given_Gbar = P_GWdata_given_Gbar

        self.H_0_pdf_single_event = P_GWdata_given_G + P_GWdata_given_Gbar

        self.H_0_pdf = np.product(self.H_0_pdf_single_event, axis=0)

        self.H_0_pdf /= np.sum(self.H_0_pdf) * (self.H_0_increment)

        return self.H_0_pdf

    def BB1_p(self,x,b,u,l):
        b = b-2
        C =1/((neg_gamma(1+b))*(1-1/(1+u/l)**(b+1)))
        p =(C/u)*(1-np.exp(-x/l))*((x/u)**b)*np.exp(-x/u)
        return p

    ### Better completeness

    def denominator_integrand(self, z , H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0)

    def schecher_lum_uni(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammaincc(self.SurveyAndEventData.alpha, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star )

    def schecher_lum_prop(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammaincc(self.SurveyAndEventData.alpha + 1, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) )

    #def BB1_lum_uni(self, z, H_0):
    #    return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2))) - ((4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))**(1 + self.SurveyAndEventData.beta)) * np.exp(-4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))/gamma_function(self.SurveyAndEventData.beta + 2) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta)) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) - ((4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2))))**(1 + self.SurveyAndEventData.beta)) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) / gamma_function(2 + self.SurveyAndEventData.beta) ))

    def BB1_lum_uni(self, z, H_0):
        return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2))) - ((4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))**(1 + self.SurveyAndEventData.beta)) * np.exp(-4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))*(1 - np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) )/gamma_function(self.SurveyAndEventData.beta + 2) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta)) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) ))


    #def BB1_lum_prop(self, z, H_0):
    #    return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta)) * gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) )
    
    def BB1_lum_prop(self, z, H_0):
        return (1/(1 - (1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta))) * z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) ) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-2-self.SurveyAndEventData.beta)) * gammaincc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) )
    
    def P_G_single(self, H_0_values):
        iss = np.zeros(len(H_0_values))
        for i in range(len(H_0_values)):
            iss[i] = quad(self.BB1_lum_uni, 0, np.inf, args=(H_0_values[i],))[0] / quad(self.denominator_integrand, 0, np.inf, args=(H_0_values[i],))[0]
        return iss

    # (self.SurveyAndEventData.L_star * (H_0/70)**(-2))

    def gw_denominator_schecher_lum_uni(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammainc(self.SurveyAndEventData.alpha, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star )

    def gw_numerator_schecher_lum_uni(self, z, x_GW, H_0):
        return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * gammainc(self.SurveyAndEventData.alpha, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star )

    def gw_denominator_schecher_lum_prop(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * gammainc(self.SurveyAndEventData.alpha + 1, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) )

    def gw_numerator_schecher_lum_prop(self, z, x_GW, H_0):
        return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * gammainc(self.SurveyAndEventData.alpha + 1, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) )

    #def gw_denominator_BB1_lum_uni(self, z, H_0):
    #    return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) ) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) )**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) )/gamma_function(2 + self.SurveyAndEventData.beta)) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2))))**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) / gamma_function(2 + self.SurveyAndEventData.beta) ))

    #def gw_numerator_BB1_lum_uni(self, z, x_GW, H_0):
    #    return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * ( (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2))) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)))**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)))/gamma_function(2 + self.SurveyAndEventData.beta)) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2))))**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) / gamma_function(2 + self.SurveyAndEventData.beta) ))

    def gw_denominator_BB1_lum_uni(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2))) + 
                ((4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))**(1 + self.SurveyAndEventData.beta)) * np.exp(-4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))*(1 - np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) )/gamma_function(self.SurveyAndEventData.beta + 2) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta)) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) ))
    
    def gw_numerator_BB1_lum_uni(self, z, x_GW, H_0):
        return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2))) + 
                ((4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))**(1 + self.SurveyAndEventData.beta)) * np.exp(-4 * np.pi * self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)))*(1 - np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) )/gamma_function(self.SurveyAndEventData.beta + 2) - ((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(-1-self.SurveyAndEventData.beta)) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *(1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))) ))

    #####def gw_denominator_BB1_lum_uni(self, z, H_0):
    #    return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * ( (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)/gamma_function(2 + self.SurveyAndEventData.beta)) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum))**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) / gamma_function(2 + self.SurveyAndEventData.beta) ))

    #####def gw_numerator_BB1_lum_uni(self, z, x_GW, H_0):
    #    return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * ( (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / self.SurveyAndEventData.L_star)/gamma_function(2 + self.SurveyAndEventData.beta)) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * ( gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) + (4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum))**(1 + self.SurveyAndEventData.beta) * np.exp(- 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/self.SurveyAndEventData.L_star + 1/self.SurveyAndEventData.min_lum)) / gamma_function(2 + self.SurveyAndEventData.beta) ))



    def gw_denominator_BB1_lum_prop(self, z, H_0):
        return z**2 * self.burr_cdf_x(self.SurveyAndEventData.max_D, self.c * z / H_0) * (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) ) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))))

    def gw_numerator_BB1_lum_prop(self, z, x_GW, H_0):
        return z**2 * self.burr_pdf(x_GW, self.c * z / H_0) * (gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 / (self.SurveyAndEventData.L_star * (H_0/70)**(-2)) ) - (1/((1 + self.SurveyAndEventData.L_star/self.SurveyAndEventData.min_lum)**(2 + self.SurveyAndEventData.beta))) * gammainc(self.SurveyAndEventData.beta + 2, 4*np.pi*self.SurveyAndEventData.min_flux * (self.c * z / H_0)**2 *( 1/(self.SurveyAndEventData.L_star * (H_0/70)**(-2)) + 1/(self.SurveyAndEventData.min_lum * (H_0/70)**(-2)))))



    '''
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

    '''



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
