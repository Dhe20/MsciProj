#%%
import numpy as np
from EventGenerator import EventGenerator
from SurveyAndEventData import SurveyAndEventData
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%

class Inference(SurveyAndEventData):
    def __init__(self, SurveyAndEventData, resolution_H_0=100):
        self.SurveyAndEventData = SurveyAndEventData
        self.distribution_calculated = False
        self.resolution_H_0 = resolution_H_0
    
    def H_0_Prob(self):
        self.distribution_calculated = True

        self.H_0_pdf = np.zeros(self.resolution_H_0)
        self.H_0_range = np.linspace(50, 100, self.resolution_H_0)
        if self.SurveyAndEventData.dimension == 2:
            self.H_0_pdf = self.H_0_inference_2d_perfect_survey()
        if self.SurveyAndEventData.dimension == 3:
            self.H_0_pdf = self.H_0_inference_3d_perfect_survey()

        return self.H_0_pdf

    def H_0_inference_2d_perfect_survey(self):
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
                    H_0_pdf_slice_single_event += D * self.SurveyAndEventData.fluxes[g] * u_r * self.SurveyAndEventData.burr(u_r,
                                                                                    self.SurveyAndEventData.BVM_c,
                                                                                    self.SurveyAndEventData.BVM_k,
                                                                                    D) * self.SurveyAndEventData.von_misses(u_phi, 
                                                                                    phi, self.SurveyAndEventData.BVM_kappa) 
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event   
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event/np.sum(H_0_pdf_single_event)
            else:
                self.H_0_pdf *= H_0_pdf_single_event/np.sum(H_0_pdf_single_event)
        return self.H_0_pdf

    def H_0_inference_3d_perfect_survey(self):
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
                    H_0_pdf_slice_single_event += (D**2) * self.SurveyAndEventData.fluxes[g] *(u_r**2) * self.SurveyAndEventData.burr(u_r,
                                                                                    self.SurveyAndEventData.BVM_c,
                                                                                    self.SurveyAndEventData.BVM_k,
                                                                                    D) * self.SurveyAndEventData.von_misses_fisher(
                                                                                    phi, u_phi, theta, u_theta, self.SurveyAndEventData.BVM_kappa)
                H_0_pdf_single_event[H_0_index] += H_0_pdf_slice_single_event   
            if event_num == 0:
                self.H_0_pdf += H_0_pdf_single_event/np.sum(H_0_pdf_single_event)
            else:
                self.H_0_pdf *= H_0_pdf_single_event/np.sum(H_0_pdf_single_event)
        return self.H_0_pdf

    def plot_H_0(self):
        if not self.distribution_calculated:
            self.H_0_Prob()
        plt.plot(self.H_0_range, self.H_0_pdf/np.sum(self.H_0_pdf))
        plt.axvline(x=70, c='r', ls='--')
        plt.show()

#%%

Gen = EventGenerator(dimension = 2, size = 50, event_count=10,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Clustered",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=1000,
                     event_distribution="Proportional", noise_distribution = "BVM", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=500)
#Gen.plot_universe_and_events()
Data = Gen.GetSurveyAndEventData()
Y = Inference(Data)
Y.plot_H_0()









# %%
