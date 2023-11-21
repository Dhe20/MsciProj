from EventGenerator import EventGenerator
from Inference import Inference

Gen = EventGenerator(dimension = 2, size = 50, event_count=20,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=25,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=False)
Gen.plot_universe_and_events()
# Data = Gen.GetSurveyAndEventData()
# Y = Inference(Data)
# Y.plot_H_0()
