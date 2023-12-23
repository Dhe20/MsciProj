from mayavi import mlab
from traits.api import HasTraits, Range, Instance,on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
from Components.EventGenerator import EventGenerator
import numpy as np
from scipy import interpolate

class Sliding_Universe_3d(HasTraits):
    slider = Range(40, 140, 70, )
    # figure = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(600, 600))
    scene = Instance(MlabSceneModel, ())

    def __init__(self, Gen):
        self.Gen = Gen
        HasTraits.__init__(self)

        self.x = self.Gen.detected_coords[:, 0]
        self.y = self.Gen.detected_coords[:, 1]
        self.z = self.Gen.detected_coords[:, 2]

        self.Rs = ((self.x ** 2) + (self.y ** 2) + (self.z ** 2))**(1/2)
        self.phis = np.arctan2(self.y, self.x)
        self.thetas = np.arccos(self.z/self.Rs)


        self.s = mlab.points3d(self.x, self.y, self.z, self.Gen.detected_luminosities ** (1 / 3), figure=self.scene.mayavi_scene,
                               scale_factor = 1, color = (1,1,1),  mode='sphere')

        mlab.points3d(0, 0, 0, color=(1.0, 1.0, 1.0), mode='axes', scale_factor=self.Gen.size / 20, figure=self.scene.mayavi_scene)

        for i, PDF in enumerate(self.Gen.BH_detected_meshgrid):
            X, Y, Z = self.Gen.BH_contour_meshgrid
            n = 1000
            PDF = PDF / PDF.sum()
            t = np.linspace(0, PDF.max(), n)
            integral = (((PDF >= t[:, None, None, None]) * PDF).sum(axis=(1, 2, 3)))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))

            mlab.contour3d(X, Y, Z, PDF, contours=[*t_contours], opacity=0.3, colormap="RdBu", figure=self.scene.mayavi_scene)

        mlab.points3d(0, 0, 0, self.Gen.max_D * 2, color=(1.0, 1.0, 1.0), mode='sphere', opacity=0.05, scale_factor=1, figure=self.scene.mayavi_scene)
        mlab.outline(
            extent=[-self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size],)

        self.scene.background = (0,0,0)

    @on_trait_change('slider')
    def slider_changed(self):
        coords = self.get_coords(self.slider)
        self.s.mlab_source.trait_set(x=coords[:,0], y = coords[:,1], z =coords[:,2])

    def Phis2Coords(self, Rs):
        Points = np.zeros((len(self.phis), 3))

        Points[:, 0] = Rs * np.cos(self.phis) * np.sin(self.thetas)
        Points[:, 1] = Rs * np.sin(self.phis) * np.sin(self.thetas)
        Points[:, 2] = Rs * np.cos(self.thetas)
        return Points

    def get_coords(self, H_0):
        Rs = self.Gen.detected_redshifts / H_0
        coords = self.Phis2Coords(Rs)
        return coords

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=600,
                     width=600
                     ),
                Group("slider"),
                )
