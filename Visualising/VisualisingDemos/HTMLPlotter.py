import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec, collections
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.widgets import Button, Slider, RangeSlider
import seaborn as sns
import matplotlib.patches as mpatches
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpld3
from mpld3 import plugins
import pickle
from Visualising.VisualisingDemos.SliderView import SliderView

H_0_resolution = 400

# Replace 'path_to_your_file/memorised_coords.csv' with the actual path to your CSV file
file_path = 'memorised_coords.csv'

# Read the CSV file into a NumPy array
coords_arr = np.loadtxt(file_path, delimiter=',')

with open('Generator.pickle', 'rb') as pickle_file:
    Gen = pickle.load(pickle_file)

with open('Inference.pickle', 'rb') as pickle_file:
    I = pickle.load(pickle_file)

filler = None
single_filler = None
collection = None
H_0_slider = None
# event_slider = None
button = None
ax1 = None
ax2 = None
ax3 = None
fig = None
colors = None



H_0_Pdf = I.H_0_Prob()


colors = sns.color_palette(n_colors=len(Gen.BH_true_luminosities))

# Create the figure and the line that we will manipulate
fig = plt.figure()

# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(9)
#
# set width of each subplot as 8
fig.set_figwidth(9*4/3)

# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         height_ratios=[4, 1, 1], wspace=0.2,
                         hspace=0.2)

ax1 = fig.add_subplot(spec[0], aspect='equal')
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])







for i, Z in enumerate(Gen.BH_detected_meshgrid):
    Xgrid, Ygrid = Gen.BH_contour_meshgrid
    z = Z
    n = 1000
    z = z / z.sum()
    t = np.linspace(0, z.max(), n)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
    ax1.contour(Xgrid, Ygrid, z, t_contours, colors=colors[i])

xhat, yhat = zip(*Gen.BH_detected_coords)
for (xhat, yhat, s) in zip(xhat, yhat, Gen.BH_true_luminosities):
    ax1.add_artist(plt.Circle(xy=(xhat, yhat), radius=s*Gen.size/100 + 0.001 * Gen.L_star, color="r", zorder=4))

ax1.add_patch(plt.Circle((0, 0), Gen.max_D, color='w', ls="--", fill=""))

H_0_s = np.linspace(50,100,H_0_resolution)
coord_indices_start = np.where(H_0_s-70 < 0 )[0][-1]
prior_H_0 = coord_indices_start
coords_start = coords_arr[coord_indices_start:coord_indices_start+2]

collection_list = []
for i in range(len(coords_arr)//2):
    patches = []
    x = coords_arr[2*i, :]
    y = coords_arr[2*i +1, :]
    for (x, y, s) in zip(x, y, Gen.detected_luminosities):
        if i == coord_indices_start:
            patches.append(mpatches.Circle(xy=(x, y), radius=s*Gen.size/100 + 0.001 * Gen.L_star, color="w", zorder=3, alpha = 1))
        else:
            patches.append(mpatches.Circle(xy=(x, y), radius=s *Gen.size/100 + 0.001 * Gen.L_star, color="r", zorder=3, alpha = 1))

    collection_per_H_0 = collections.PatchCollection(patches, match_original=True)
    collection_list.append(collection_per_H_0)

collection = collection_list[coord_indices_start]
line = ax1.add_collection(collection)
ax1.set_ylim(-Gen.size, Gen.size)
ax1.set_xlim(-Gen.size, Gen.size)

ax2.plot(I.H_0_range, H_0_Pdf, c='w', label='Combined H_0 Posterior')

red_patch = mpatches.Patch(color='none', label='Single Event Posteriors')

for i, Single_H_0_Pdf in enumerate(I.H_0_pdf_single_event):
    ax3.plot(I.H_0_range, Single_H_0_Pdf, c=colors[i])
ax3.legend(handles=[red_patch], fontsize=8)

filler = ax2.fill_between(
    [I.H_0_range[np.where(I.H_0_range < 70)[0][-1]], I.H_0_range[np.where(I.H_0_range > 70)[0][0]]],
    [H_0_Pdf[np.where(I.H_0_range < 70)[0][-1]], H_0_Pdf[np.where(I.H_0_range > 70)[0][0]]],
    alpha=0.7, color='w')

single_color = colors[np.where(
    I.H_0_pdf_single_event == np.max(I.H_0_pdf_single_event[:, [np.where(I.H_0_range < 70)[0][-1]]]))[0][
    0]]

single_filler = ax3.fill_between(
    [I.H_0_range[np.where(I.H_0_range < 70)[0][-1]], I.H_0_range[np.where(I.H_0_range > 70)[0][0]]],
    [np.max(I.H_0_pdf_single_event[:, [np.where(I.H_0_range < 70)[0][-1]]]),
     np.max(I.H_0_pdf_single_event[:, [np.where(I.H_0_range > 70)[0][0]]])], alpha=0.7, color=single_color)

ax2.legend(fontsize=8)
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)


def update(val):
    global collection_list, collection, line

    H_0 = H_0_slider.val
    H_0_index = np.where(H_0_s == H_0)[0][0]

    collection.remove()
    collection = collection_list[H_0_index]
    line = ax1.add_collection(collection)
    return line
    # fig.canvas.draw_idle()

#
# # adjust the main plot to make room for the sliders
# fig.subplots_adjust(bottom=0.2)
#
# # Make a horizontal slider to control the frequency.
#
#

# register the update function with each slider

# event_slider.on_changed(update_events)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.

# resetax = fig.add_axes([0.8, 0.025, 0.1, 0.03])
# button = Button(resetax, 'Reset', hovercolor='0.975')
#

#
# # button.on_clicked(reset_H_0)
#


# fig.subplots_adjust(bottom=0.2)
# axH_0 = fig.add_axes([0.2, 0.1, 0.65, 0.03])
# H_0_slider = Slider(
#     ax=axH_0,
#     label=r'$H_{0}$',
#     valmin=H_0_s[0],
#     valmax=H_0_s[-1],
#     valinit=H_0_s[coord_indices_start],
#     valstep=H_0_s,
#     initcolor='none'
# )
# H_0_slider.on_changed(update)
# plt.show()
mpld3.plugins.connect(fig, SliderView(line, callback_func="update"))
mpld3.show()