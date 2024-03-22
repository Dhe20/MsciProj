##

from Components.EventGenerator import  EventGenerator
from Components.Inference import Inference
from mayavi import mlab
import moviepy.editor as mpy
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.03*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=5, total_luminosity=500,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=100, plot_contours=True, seed = 22)
Data = Gen.GetSurveyAndEventData()
print(Gen.detected_event_count)

fig = Gen.plot_universe_and_events(show = False)

ts = np.linspace(0,5,200)
def make_frame(t):
    # camera angle
    mlab.view(azimuth=360 * t / duration, elevation=-70, distance=360, focalpoint=[0,0,0])
    mlab.savefig("Images/" + f"frame_{t:03}.jpg")
    return None


duration = 5
# make_frame(t=0)

for t in tqdm(ts):
    make_frame(t=t)
    # imageio.imwrite("/Users/daneverett/PycharmProjects/MSciProject/Visualising/3DGIF/Images/"+str(t)+'_screenshot.png', arr)

# Save the image array as a PNG file using imageio


# animation = mpy.VideoClip(make_frame, duration=duration).resize(0.5)
# # Video generation takes 10 seconds, GIF generation takes 25s
# animation.write_videofile("Rotated_sym_distri_static_Python.mp4", fps=20)
# animation.write_gif("Rotated_3d_universe.gif", fps=20)
# mlab.close(fig)

##

import os
import imageio


def create_gif_from_images(folder_path, output_gif_name, duration=0.5):
    """
    Create a GIF from numbered images in a folder.

    Args:
    - folder_path (str): Path to the folder containing the images.
    - output_gif_name (str): Name of the output GIF file (including .gif extension).
    - duration (float): Duration of each frame in the GIF, in seconds.
    """
    images = []
    # List all files in the folder and sort them
    file_names = sorted((fn for fn in os.listdir(folder_path) if fn.startswith('frame_') and fn.endswith('.jpg')),
                        key=lambda x: float(x.split('_')[1].split('.jpg')[0]))

    # Read each file and append to images list
    for filename in file_names:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))

    # Save the images as a GIF
    imageio.mimsave(output_gif_name, images, duration=duration)


# Example usage:
folder_path = '/Users/daneverett/PycharmProjects/MSciProject/Visualising/3DGIF/Images'
output_gif_name = '3D_Spinning.gif'
create_gif_from_images(folder_path, output_gif_name, duration=duration)