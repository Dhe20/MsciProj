from traits.api import HasTraits, Range, Instance,on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

#
# from EventGenerator import EventGenerator
from mayavi import mlab
import numpy as np
from scipy import interpolate


import numpy as np
from mayavi import mlab

from traits.api import HasTraits, Range, Instance,on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel



class MyModel(HasTraits):
    slider    = Range(-5., 5., 0.5, )
    scene = Instance(MlabSceneModel, ())

    def __init__(self, Gen):
        HasTraits.__init__(self)
        self.s = mlab.points3d(x, y, z, figure=self.scene.mayavi_scene)

    @on_trait_change('slider')
    def slider_changed(self):
        self.s.mlab_source.trait_set(x=x*self.slider)
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene)),
                Group("slider"))

my_model = MyModel()
my_model.configure_traits()