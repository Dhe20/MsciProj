import imageio

# Define the file pattern for the saved images
file_pattern = 'Images/figure_cluster_coeff_{}.png'
# Define the duration that each image (frame) will display in the GIF
frame_duration = 2 # seconds per frame

# Create a list to hold the images
images = []

# Load each file into the images list
for i in range(0, 11):  # Assuming your cluster coefficients start from 1
    filename = file_pattern.format(i)
    images.append(imageio.imread(filename))

# Save the images as a GIF
output_gif_path = 'cluster_coeff_animation.gif'
imageio.mimsave(output_gif_path, images, duration=frame_duration)