from EventGenerator import EventGenerator
from mayavi import mlab
import numpy as np
from tvtk.tools import visual
from scipy.interpolate import griddata
import numpy as np
import cmath
from scipy import interpolate



size = 50

Gen = EventGenerator(dimension = 3, size = size, event_count=5,
                     luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=1, total_luminosity=100,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=75, plot_contours=True)

N=10

x = Gen.detected_coords[:,0]
y = Gen.detected_coords[:,1]
z = Gen.detected_coords[:,2]








fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size = (600,600))


mlab.points3d(0,0,0, color = (1.0,1.0,1.0), mode ='axes', scale_factor = size/10)
mlab.points3d(x,y,z, Gen.detected_luminosities**(1/3) , color = (1.0,1.0,1.0), mode='sphere', opacity = 1, scale_factor = 1)
for (xhat, yhat, zhat, s) in zip(*zip(*Gen.BH_true_coords), Gen.BH_true_luminosities**(1/3)):
    mlab.points3d(xhat, yhat, zhat, s*1.01, color=(0.0,1.0,0.0), mode='sphere', opacity=1, scale_factor = 1)
for (xhat, yhat, zhat, s) in zip(*zip(*Gen.BH_detected_coords), Gen.BH_detected_luminosities**(1/3)):
    mlab.points3d(xhat, yhat, zhat, s, color=(1.0,0.0,0.0), mode='sphere', opacity=1, scale_factor = 1)
mlab.points3d(0,0,0, Gen.max_D*2, color = (1.0,1.0,1.0), mode ='sphere', opacity = 0.05, scale_factor = 1)

for i, PDF in enumerate(Gen.BH_detected_meshgrid):
    X, Y, Z = Gen.BH_contour_meshgrid
    n = 1000
    PDF = PDF / PDF.sum()
    t = np.linspace(0, PDF.max(), n)
    integral = (((PDF >= t[:, None, None, None]) * PDF).sum(axis=(1, 2, 3)))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([0.9973, 0.9545, 0.6827]))

    mlab.contour3d(X, Y, Z, PDF, contours=[*t_contours], opacity=0.15, colormap = "RdBu")


# mlab.contour3d(X.T, Y.T, Z.T, PDF)
mlab.outline(extent = [-size, size, -size, size, -size, size])
mlab.show()

