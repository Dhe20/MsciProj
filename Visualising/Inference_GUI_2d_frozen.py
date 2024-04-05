from Components.EventGenerator import EventGenerator
from Components.Inference import Inference
import numpy as np
from matplotlib import gridspec, collections
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.widgets import Button, Slider, RangeSlider
import seaborn as sns
import matplotlib.patches as mpatches
import sys


class InferenceGUI:
    def __init__(self, Inference_Object, Data_Object, Generator_Object, H_0 = 70):

        self.H_0 = H_0
        self.Data = Data_Object
        self.Gen = Generator_Object
        self.I = Inference_Object


        self.H_0_Pdf = self.I.H_0_Prob()
        self.Phis = np.arctan2(self.Data.detected_coords[:,1], self.Data.detected_coords[:,0])
        self.filler = None
        self.single_filler = None
        self.collection = None
        self.H_0_slider = None
        # self.event_slider = None
        self.button = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.fig = None
        self.colors = None
        self.fig2 = None



    def Phis2Coords(self, Rs):
        Points = np.zeros((len(self.Phis), 2))
        Points[:,0] = Rs*np.cos(self.Phis)
        Points[:,1] = Rs*np.sin(self.Phis)
        return Points

    def get_coords(self, H_0):
        Rs = self.Data.detected_redshifts*self.Gen.c/H_0
        coords = self.Phis2Coords(Rs)
        return coords

    def reset_H_0(self, event):
        self.H_0_slider.reset()
        self.ax1.set_ylim(-self.Gen.size,self.Gen.size)
        self.ax1.set_xlim(-self.Gen.size,self.Gen.size)

    def view(self):

        self.colors = sns.color_palette("bright", n_colors = len(self.Gen.BH_true_luminosities))

        # Create the figure and the line that we will manipulate
        self.fig = plt.figure()


        # to change size of subplot's
        # set height of each subplot as 8
        self.fig.set_figheight(8)

        # set width of each subplot as 8
        self.fig.set_figwidth(8)

        self.ax1 = self.fig.add_subplot(aspect='equal')


        for i, Z in enumerate(self.Gen.BH_detected_meshgrid):
            Xgrid, Ygrid = self.Gen.BH_contour_meshgrid
            z = Z
            n = 1000
            z = z / z.sum()
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
            self.ax1.contour(Xgrid, Ygrid, z, t_contours, colors=self.colors[i])

        xhat, yhat = zip(*self.Gen.BH_detected_coords)
        for (xhat, yhat, s) in zip(xhat, yhat, self.Gen.BH_true_luminosities):
            self.ax1.add_artist(mpatches.Circle(xy=(xhat, yhat), radius=self.Gen.L_star, color="r", zorder=4))

        self.ax1.add_patch(mpatches.Circle((0, 0), self.Gen.max_D, color='w', ls="--", fill=""))

        coords = self.get_coords(H_0 = self.H_0)

        x = coords[:,0]
        y = coords[:,1]
        patches = []
        for (x, y, s) in zip(x, y, self.Gen.detected_luminosities):
            patches.append(mpatches.Circle(xy=(x, y), radius=s + 0.001 * self.Gen.L_star, color="w", zorder=3))
        self.collection = collections.PatchCollection(patches, alpha=1, match_original=True)
        self.ax1.add_collection(self.collection)
        self.ax1.set_ylim(-self.Gen.size,self.Gen.size)
        self.ax1.set_xlim(-self.Gen.size,self.Gen.size)

        self.ax1.axes.get_xaxis().set_visible(False)
        self.ax1.axes.get_yaxis().set_visible(False)

        if self.H_0 % 1 == 0:
            self.H_0_str = round(self.H_0)
        else:
            self.H_0_str = str(self.H_0)
            self.H_0_str = self.H_0_str[0:2] + "_" + self.H_0_str[-1]

        xlim = [-110, 15]
        ylim = [130, 310]

        x, y, height = 0.01, 0.5, 0.49
        width = height * (xlim[-1] - xlim[0]) / (ylim[-1] - ylim[0])

        inset_ax = self.ax1.inset_axes(
            [x, y, width, height],  # [x, y, width, height] w.r.t. axes
            xlim=xlim, ylim=ylim,  # sets viewport & tells relation to main axes
            xticklabels=[], yticklabels=[]
        )

        coord = np.array([-30, 288])
        index = np.argmin(np.sum(np.square(self.Gen.detected_coords - coord), axis=1))
        true_coord = self.Gen.detected_coords[index]

        x, y = zip(*self.Gen.detected_coords)
        cutoff = plt.Circle((0, 0), self.Gen.max_D, color='w', ls="--", fill="")
        inset_ax.add_patch(cutoff)
        for _ in range(self.Gen.n):
            inset_ax.plot(self.Gen.distance_range[_, :, 0], self.Gen.distance_range[_, :, 1], "-", color="b", )

        coords = self.get_coords(H_0 = self.H_0)

        x = coords[:,0]
        y = coords[:,1]

        for i, (x, y, s) in enumerate(zip(x, y, self.Gen.detected_luminosities)):
            if i == index:
                neighbouring_galaxy = plt.Circle(xy=(x, y), radius=s + 0.001 * self.Gen.L_star, color="white", zorder=3)
            inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s + 0.001 * self.Gen.L_star, color="white", zorder=3))
        inset_ax.scatter(0, 0, s=self.Gen.size / 1.25, c="w", marker="x")
        inset_ax.xaxis.set_tick_params(labelbottom=False)
        inset_ax.yaxis.set_tick_params(labelleft=False)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        plt.tight_layout()

        for (xhat, yhat, s) in zip(*zip(*self.Gen.BH_detected_coords), self.Gen.BH_detected_luminosities):
            inset_ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=self.Gen.L_star, color="red", zorder=4))
            detected_patch = plt.Circle(xy=(xhat, yhat), radius=self.Gen.L_star, color="red", zorder=4)

        for i, Z in enumerate(self.Gen.BH_detected_meshgrid):
            X, Y = self.Gen.BH_contour_meshgrid
            z = Z
            n = 1000
            z = z / z.sum()
            t = np.linspace(0, z.max(), n)
            integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
            inset_ax.contour(X, Y, z, t_contours, colors=self.colors[-1], zorder=2)

        # x, y = zip(*self.Gen.BH_true_coords)
        # for (x, y, s) in zip(x, y, self.Gen.BH_true_luminosities):
        #     inset_ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="cyan", zorder=4))
        #     true_patch = plt.Circle(xy=(x, y), radius=s, color="cyan", zorder=4)

        self.ax1.indicate_inset_zoom(inset_ax, edgecolor="white")

        self.fig.savefig(
            "/Users/daneverett/PycharmProjects/MSciProject/Visualising/HTMLPlot/images/range-slider/new_top_" + str(
                self.H_0_str) + ".jpg", dpi=200)
        plt.close(self.fig)
        # plt.show()

        self.fig2 = plt.figure()

        # to change size of subplot's
        # set height of each subplot as 8
        self.fig2.set_figheight(8 * 2 / 6)

        # set width of each subplot as 8
        self.fig2.set_figwidth(8)

        # # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                                 height_ratios=[1, 1], wspace=0.2,
                                 hspace=0.2)

        self.ax2 = self.fig2.add_subplot(spec[0])
        self.ax3 = self.fig2.add_subplot(spec[1], sharex=self.ax2)

        self.ax2.plot(self.I.H_0_range, self.H_0_Pdf, c='w', label = 'Combined H_0 Posterior')
        plt.setp(self.ax2.get_xticklabels(), visible=False)

        red_patch = mpatches.Patch(color='none', label='Single Event Posteriors')


        for i, Single_H_0_Pdf in enumerate(self.I.H_0_pdf_single_event):
            self.ax3.plot(self.I.H_0_range, (10**4)*Single_H_0_Pdf, c = self.colors[i])

        self.ax3.legend(handles=[red_patch], fontsize=8)

        self.filler = self.ax2.fill_between([self.I.H_0_range[np.where(self.I.H_0_range < self.H_0)[0][-1]], self.I.H_0_range[np.where(self.I.H_0_range > self.H_0)[0][0]]],
                        [self.H_0_Pdf[np.where(self.I.H_0_range < self.H_0)[0][-1]], self.H_0_Pdf[np.where(self.I.H_0_range > self.H_0)[0][0]]], alpha=0.7, color='w')

        single_color = self.colors[np.where(self.I.H_0_pdf_single_event == np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range < self.H_0)[0][-1]]]))[0][0]]

        self.single_filler = self.ax3.fill_between([self.I.H_0_range[np.where(self.I.H_0_range < self.H_0)[0][-1]], self.I.H_0_range[np.where(self.I.H_0_range > self.H_0)[0][0]]],
                        [np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range < self.H_0)[0][-1]]])*10**4, np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range > self.H_0)[0][0]]]*10**4)], alpha=0.7, color=single_color)

        # adjust the main plot to make room for the sliders

        # Make a horizontal slider to control the frequency.


        # axEvent = self.fig.add_axes([0.2, 0.55, 0.03, 0.3])
        # self.event_slider = RangeSlider(
        #     ax=axEvent,
        #     label='Events Shown',
        #     valmin=1,
        #     valmax=self.Data.detected_event_count,
        #     valinit=(0, self.Data.detected_event_count),
        #     valstep=np.arange(1,self.Data.detected_event_count+1),
        #     orientation = 'vertical'
        # )
        #

        # The function to be called anytime a slider's value changes

        # register the update function with each slider

        # self.event_slider.on_changed(self.update_events)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.

        self.ax2.legend(fontsize=8)
        self.fig2.savefig("/Users/daneverett/PycharmProjects/MSciProject/Visualising/HTMLPlot/images/range-slider/new_bottom_" + str(self.H_0_str) + ".jpg", dpi=100)
        plt.close(self.fig2)
        # self.button.on_clicked(self.reset_H_0)


        # plt.show()


    # def update_events(self, val):
    #     lower = self.event_slider.val[0]
    #     upper = self.event_slider.val[1]
