import numpy as np
class SurveyAndEventData:
    def __init__(self, dimension, detected_coords, detected_luminosities,
                 detected_redshifts, detected_redshifts_uncertainties, fluxes, BH_detected_coords, BVM_k,
                 BVM_c, BVM_kappa, BurrFunc, VonMissesFunc, VonMissesFisherFunc, event_distribution,
                 contour_type, noise_distribution, noise_sigma, max_D, redshift_noise_sigma,
                 detected_event_count, sample_time, c, min_flux = 0, event_rate = 0 ):
        self.min_flux = min_flux

        self.detected_galaxy_indices = np.where(fluxes > self.min_flux)[0]

        self.detected_coords = detected_coords[self.detected_galaxy_indices]
        self.detected_redshifts = detected_redshifts[self.detected_galaxy_indices]
        self.detected_redshifts_uncertainties = detected_redshifts_uncertainties[self.detected_galaxy_indices]
        self.detected_luminsoties = detected_luminosities[self.detected_galaxy_indices]
        self.fluxes = fluxes[self.detected_galaxy_indices]
        
        self.dimension = dimension
        self.c = c

        self.BH_detected_coords = BH_detected_coords
        self.contour_type = contour_type
        self.noise_distribution = noise_distribution
        self.noise_sigma = noise_sigma
        self.redshift_noise_sigma = redshift_noise_sigma
        self.BVM_k = BVM_k
        self.BVM_c = BVM_c
        self.BVM_kappa = BVM_kappa
        self.burr = BurrFunc
        self.von_misses = VonMissesFunc
        self.von_misses_fisher = VonMissesFisherFunc
        self.max_D = max_D
        self.detected_event_count = detected_event_count
        self.gamma_upper_lim = 10**(12)
        self.gamma_lower_lim = 10
        self.sample_time = sample_time
        self.min_flux = min_flux
        #For evaluating effect of Gamma
        self.event_rate = event_rate
        self.event_distribution = event_distribution
