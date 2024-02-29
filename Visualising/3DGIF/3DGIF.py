from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from mayavi import mlab

import time
Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.0*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=.5, total_luminosity=10,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=100, plot_contours=True, seed = 42)
Data = Gen.GetSurveyAndEventData()
# print(Gen.detected_event_count)
# print(len(Gen.detected_redshifts))

fig = Gen.plot_universe_and_events(show = False)
camera = fig.scene.camera




# # Rotate the scene
for _ in range(360): # Rotate for 360 degrees as an example
#     # _ = _*36
# #     mlab.view(azimuth=36, elevation=None, distance=None, figure=fig, roll=None) # rotates the scene by 1 degree along the azimuth
# #     mlab.draw(fig) # Redraw the figure to update the scene
# #     # print(f"frame_{_:03}.png")
    camera.roll(1)
    # time.sleep(1)  # Wait for 1 second
#     # mlab.savefig("Images/"+f"frame_{_:03}.jpg")
# #
mlab.show()
