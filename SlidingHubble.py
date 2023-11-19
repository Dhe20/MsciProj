from EventGenerator import EventGenerator
from Inference import Inference
import numpy as np
from matplotlib import gridspec, collections
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.widgets import Button, Slider
import seaborn as sns
import matplotlib.patches as mpatches



Gen = EventGenerator(dimension = 2, size = 50, event_count=5,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Clustered",
                     cluster_coeff=5, characteristic_luminosity=.1, total_luminosity=200,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True)

Data = Gen.GetSurveyAndEventData()
resolution = 100
H_0_Min = 40
H_0_Max = 100
Y = Inference(Data, H_0_Min = H_0_Min, H_0_Max = H_0_Max, resolution_H_0 = resolution)

class InferenceGUI:
    def __init__(self, Inference_Object, Data_Object, Generator_Object):
        self.Data = Data_Object
        self.Gen = Generator_Object
        self.I = Inference_Object


        self.H_0_Pdf = self.I.H_0_Prob()
        self.Phis = np.arctan2(self.Data.detected_coords[:,1], self.Data.detected_coords[:,0])
        self.filler = None
        self.single_filler = None
        self.collection = None
        self.H_0_slider = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.fig = None
        self.colors = None


    def Phis2Coords(self, Rs):
        Points = np.zeros((len(self.Phis), 2))
        Points[:,0] = Rs*np.cos(self.Phis)
        Points[:,1] = Rs*np.sin(self.Phis)
        return Points

    def get_coords(self, H_0):
        Rs = self.Data.detected_redshifts/H_0
        coords = self.Phis2Coords(Rs)
        return coords

    def View(self):

        self.colors = sns.color_palette(n_colors = len(self.Gen.BH_true_luminosities))

        # Create the figure and the line that we will manipulate
        self.fig = plt.figure()


        # to change size of subplot's
        # set height of each subplot as 8
        self.fig.set_figheight(8)

        # set width of each subplot as 8
        self.fig.set_figwidth(8)

        # create grid for different subplots
        spec = gridspec.GridSpec(ncols=1, nrows=3,
                                 height_ratios=[4, 1, 1] ,wspace=0.2,
                                 hspace=0.2)

        self.ax1 = self.fig.add_subplot(spec[0], aspect='equal')
        self.ax2 = self.fig.add_subplot(spec[1])
        self.ax3 = self.fig.add_subplot(spec[2])

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
            self.ax1.add_artist(plt.Circle(xy=(xhat, yhat), radius=s, color="r", zorder=4))

        self.ax1.add_patch(plt.Circle((0, 0), self.Gen.max_D, color='w', ls="--", fill=""))



        coords = self.get_coords(H_0 = 70)

        x = coords[:,0]
        y = coords[:,1]
        patches = []
        for (x, y, s) in zip(x, y, self.Gen.detected_luminosities):
            patches.append(plt.Circle(xy=(x, y), radius=s + 0.001 * self.Gen.L_star, color="white", zorder=3))
        self.collection = collections.PatchCollection(patches, alpha=1)
        self.ax1.add_collection(self.collection)
        self.ax1.set_ylim(-self.Gen.size,self.Gen.size)
        self.ax1.set_xlim(-self.Gen.size,self.Gen.size)

        self.ax2.plot(self.I.H_0_range, self.H_0_Pdf, c='w', label = 'Combined H_0 Posterior')

        red_patch = mpatches.Patch(color='none', label='Single Event Posteriors')


        for i, Single_H_0_Pdf in enumerate(self.I.H_0_pdf_single_event):
            self.ax3.plot(self.I.H_0_range, Single_H_0_Pdf, c = self.colors[i])
        self.ax3.legend(handles=[red_patch], fontsize=8)

        self.filler = self.ax2.fill_between([self.I.H_0_range[np.where(self.I.H_0_range < 70)[0][-1]], self.I.H_0_range[np.where(self.I.H_0_range > 70)[0][0]]],
                        [self.H_0_Pdf[np.where(self.I.H_0_range < 70)[0][-1]], self.H_0_Pdf[np.where(self.I.H_0_range > 70)[0][0]]], alpha=0.7, color='w')

        single_color = self.colors[np.where(self.I.H_0_pdf_single_event == np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range < 70)[0][-1]]]))[0][0]]

        self.single_filler = self.ax3.fill_between([self.I.H_0_range[np.where(self.I.H_0_range < 70)[0][-1]], self.I.H_0_range[np.where(self.I.H_0_range > 70)[0][0]]],
                        [np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range < 70)[0][-1]]]), np.max(self.I.H_0_pdf_single_event[:,[np.where(self.I.H_0_range > 70)[0][0]]])], alpha=0.7, color=single_color)

        # adjust the main plot to make room for the sliders
        self.fig.subplots_adjust(bottom=0.25)

        # Make a horizontal slider to control the frequency.
        axH_0 = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.H_0_slider = Slider(
            ax=axH_0,
            label='H_0',
            valmin=min(self.I.H_0_range),
            valmax=max(self.I.H_0_range)-1,
            valinit=70,
            valstep=np.linspace(H_0_Min, H_0_Max, resolution),
            initcolor='none'
        )

        # The function to be called anytime a slider's value changes

        # register the update function with each slider
        self.H_0_slider.on_changed(self.update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        button.on_clicked(self.reset)
        #
        self.ax2.legend(fontsize=8)
        self.ax1.axes.get_xaxis().set_visible(False)
        self.ax1.axes.get_yaxis().set_visible(False)
        plt.show()

    def update(self, val):

        H_0 = self.H_0_slider.val

        self.filler.remove()
        self.single_filler.remove()
        self.filler = self.ax2.fill_between(
            [self.I.H_0_range[np.where(self.I.H_0_range <= H_0)[0][-1]],
             self.I.H_0_range[np.where(self.I.H_0_range > H_0)[0][0]]],
            [self.H_0_Pdf[np.where(self.I.H_0_range <= H_0)[0][-1]],
             self.H_0_Pdf[np.where(self.I.H_0_range > H_0)[0][0]]], alpha=0.7, color='w')
        coords = self.get_coords(self.H_0_slider.val)

        self.single_color = self.colors[np.where(self.I.H_0_pdf_single_event == np.max(
            self.I.H_0_pdf_single_event[:, [np.where(self.I.H_0_range <= H_0)[0][-1]]]))[0][0]]

        self.single_filler = self.ax3.fill_between([self.I.H_0_range[np.where(self.I.H_0_range <= H_0)[0][-1]],
                                                     self.I.H_0_range[np.where(self.I.H_0_range > H_0)[0][0]]],
                                                   [np.max(self.I.H_0_pdf_single_event[:,
                                                   [np.where(self.I.H_0_range <= H_0)[0][-1]]]), np.max(
                                                       self.I.H_0_pdf_single_event[:,
                                                       [np.where(self.I.H_0_range > H_0)[0][0]]])], alpha=0.7,
                                                   color=self.single_color)

        self.collection.remove()

        x = coords[:, 0]
        y = coords[:, 1]
        patches = []
        for (x, y, s) in zip(x, y, self.Gen.detected_luminosities):
            patches.append(plt.Circle(xy=(x, y), radius=s + 0.001 * self.Gen.L_star, color="white", zorder=3))

        self.collection = collections.PatchCollection(patches, alpha=1)
        self.ax1.add_collection(self.collection)
        self.fig.canvas.draw_idle()

    def reset(self, event):
        self.H_0_slider.reset()


GUI = InferenceGUI(Y, Data, Gen)
GUI.View()
