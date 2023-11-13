import numpy as np
from EventGenerator import EventGenerator
from SurveyAndEventData import SurveyAndEventData
import matplotlib.pyplot as plt
from tqdm import tqdm

class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData):
        self.SurveyAndEventData = SurveyAndEventData
        self.distribution_calculated = False
    def H_0_Prob(self, resolution = 100):

        self.distribution_calculated = True

        H_0_pdf = np.zeros(resolution)
        H_0_range = np.linspace(50, 100, resolution)
        for H_0_index, H_0 in enumerate(tqdm(H_0_range)):
            if self.SurveyAndEventData.dimension == 2:
                H_0_pdf[H_0_index] = self.H_0_inference_2d_perfect_survey(H_0)
        self.H_0_range = H_0_range
        self.H_0_pdf = H_0_pdf
        return self.H_0_pdf

    def H_0_inference_2d_perfect_survey(self, H_0):
        H_0_pdf_slice = 0
        for event_num in range(len(self.SurveyAndEventData.BH_detected_coords)):
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
                H_0_pdf_slice_single_event += D * self.SurveyAndEventData.fluxes[g] * u_r * self.SurveyAndEventData.burr(u_r,
                                                                                self.SurveyAndEventData.BVM_c,
                                                                                self.SurveyAndEventData.BVM_k,
                                                                                D) * self.SurveyAndEventData.von_misses(
                                                                                u_phi, phi, self.SurveyAndEventData.BVM_kappa)
            if event_num == 0:
                H_0_pdf_slice += H_0_pdf_slice_single_event
            else:
                H_0_pdf_slice *= H_0_pdf_slice_single_event
        return H_0_pdf_slice

    def plot_H_0(self):
        if not self.distribution_calculated:
            self.H_0_Prob()
        plt.plot(self.H_0_range, self.H_0_pdf)
        plt.show()



Gen = EventGenerator(dimension = 2, size = 50, event_count=5,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Clustered",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=10,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=500)
Gen.plot_universe_and_events()
Data = Gen.GetSurveyAndEventData()
Y = Inference(Data)
Y.plot_H_0()








