import numpy as np
from EventGenerator import EventGenerator
from SurveyAndEventData import SurveyAndEventData

Gen = EventGenerator(dimension = 2, size = 50, event_count=2,
                     luminosity_gen_type = "Cut-Schecter", coord_gen_type = "Clustered",
                     cluster_coeff=50, characteristic_luminosity=.1, total_luminosity=400,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = .5)

X = Gen.SurveyAndEventData()

class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData):
        self.SurveyAndEventData = SurveyAndEventData

    def GetPrior(self):
        weight = self.SurveyAndEventData.luminosities/np.sum(self.SurveyAndEventData.luminosities)
        dirac_delta_coords = self.SurveyAndEventData.detected_coords
        PDF = None
        return PDF



I = Inference(SurveyAndEventData)
