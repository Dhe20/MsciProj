class SurveyAndEventData:
    def __init__(self, dimension, detected_coords, detected_luminosities,
                 detected_redshifts, fluxes, BH_detected_coords, BVM_k,
                 BVM_c, BVM_kappa, BurrFunc, VonMissesFunc):
        self.dimension = dimension
        self.detected_coords = detected_coords
        self.detected_redshifts = detected_redshifts
        self.detected_luminsoties = detected_luminosities
        self.fluxes = fluxes
        self.BH_detected_coords = BH_detected_coords
        self.BVM_k = BVM_k
        self.BVM_c = BVM_c
        self.BVM_kappa = BVM_kappa
        self.burr = BurrFunc
        self.von_misses = VonMissesFunc
