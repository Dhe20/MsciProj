# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
# os.environ['ETS_TOOLKIT'] = 'qt4'
# # By default, the PySide binding will be used. If you want the PyQt bindings
# # to be used, you need to set the QT_API environment variable to 'pyqt'
# os.environ['QT_API'] = 'pyqt'
# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Range, Instance, on_trait_change
from Components.EventGenerator import EventGenerator
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor
import numpy as np
from scipy import interpolate


################################################################################
#The actual visualization
class sliding_Universe_3d(HasTraits):
    scene = Instance(MlabSceneModel, ())
    slider = Range(40, 140, 70, )

    @on_trait_change('scene.activated')
    def update_plot(self):
        self.Gen = Gen
        HasTraits.__init__(self)

        self.x = self.Gen.detected_coords[:, 0]
        self.y = self.Gen.detected_coords[:, 1]
        self.z = self.Gen.detected_coords[:, 2]

        self.Rs = ((self.x ** 2) + (self.y ** 2) + (self.z ** 2)) ** (1 / 2)
        self.phis = np.arctan2(self.y, self.x)
        self.thetas = np.arccos(self.z / self.Rs)

        self.s = self.scene.mlab.points3d(self.x, self.y, self.z, self.Gen.detected_luminosities ** (1 / 3),
                               figure=self.scene.mayavi_scene,
                               scale_factor=1, color=(1, 1, 1), mode='sphere')

        self.scene.mlab.points3d(0, 0, 0, color=(1.0, 1.0, 1.0), mode='axes', scale_factor=self.Gen.size / 20,
                      figure=self.scene.mayavi_scene)

        for i, PDF in enumerate(self.Gen.BH_detected_meshgrid):
            X, Y, Z = self.Gen.BH_contour_meshgrid
            n = 1000
            PDF = PDF / PDF.sum()
            t = np.linspace(0, PDF.max(), n)
            integral = (((PDF >= t[:, None, None, None]) * PDF).sum(axis=(1, 2, 3)))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))

            self.scene.mlab.contour3d(X, Y, Z, PDF, contours=[*t_contours], opacity=0.3, colormap="RdBu",
                           figure=self.scene.mayavi_scene)

        self.scene.mlab.points3d(0, 0, 0, self.Gen.max_D * 2, color=(1.0, 1.0, 1.0), mode='sphere', opacity=0.05,
                      scale_factor=1, figure=self.scene.mayavi_scene)
        self.scene.mlab.outline(
            extent=[-self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size], )

        self.scene.background = (0, 0, 0)





    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                height=250, width=300, show_label=False),
                resizable=True # We need this to resize with the parent widget
                )


################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.sliding_Universe_3d = sliding_Universe_3d()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        # QtCore.pyqtRemoveInputHook()
        # import pdb ; pdb.set_trace()
        # QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.sliding_Universe_3d.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


if __name__ == "__main__":
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.
    Gen = EventGenerator(dimension=3, size=50, resolution=100,
                         luminosity_gen_type="Cut-Schechter", coord_gen_type="Random",
                         cluster_coeff=5, characteristic_luminosity=5, total_luminosity=100, sample_time=0.01,
                         event_rate=10,
                         event_distribution="Proportional", contour_type="BVM", redshift_noise_sigma=0.0,
                         plot_contours=True, seed=1)

    app = QtGui.QApplication.instance()
    container = QtGui.QWidget()
    container.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
    # define a "complex" layout to test the behaviour
    layout = QtGui.QGridLayout(container)

    # put some stuff around mayavi
    label_list = []
    for i in range(3):
        for j in range(3):
            if (i==1) and (j==1):continue
            label = QtGui.QLabel(container)
            label.setText("Your QWidget at (%d, %d)" % (i,j))
            label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            layout.addWidget(label, i, j)
            label_list.append(label)


    mayavi_widget = MayaviQWidget(container)

    layout.addWidget(mayavi_widget, 1, 1)
    container.show()
    window = QtGui.QMainWindow()
    window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec()