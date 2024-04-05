import imageio

# Define the file pattern for the saved images
file_pattern = '/Users/daneverett/PycharmProjects/MSciProject/Visualising/Simulation_Explanation_2/Observer_Information_Gif/Observer_Information_{}.png'
# Define the duration that each image (frame) will display in the GIF
frame_duration = 0.1 # seconds per frame

# Create a list to hold the images
images = []

# Load each file into the images list
for i in range(0, 100):  # Assuming your cluster coefficients start from 1
    filename = file_pattern.format(i)
    images.append(imageio.imread(filename))
for i in range(0, 100):  # Assuming your cluster coefficients start from 1
    filename = file_pattern.format(99-i)
    images.append(imageio.imread(filename))

# Save the images as a GIF
output_gif_path = 'Observer_Information.gif'
imageio.mimsave(output_gif_path, images, duration=frame_duration)