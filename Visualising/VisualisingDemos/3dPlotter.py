from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference

Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.03*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=100, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()
print(Gen.detected_event_count)
