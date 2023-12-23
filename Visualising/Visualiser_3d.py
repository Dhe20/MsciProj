import numpy as np
from scipy import interpolate
class Visualiser_3d:
    def __init__(self, Gen
                         ):

        self.Gen = Gen

    def plot_universe_and_events(self):
        from mayavi import mlab
        x = self.Gen.detected_coords[:, 0]
        y = self.Gen.detected_coords[:, 1]
        z = self.Gen.detected_coords[:, 2]
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(600, 600))
        # mlab.points3d(0, 0, 0, color=(1.0, 1.0, 1.0), mode='axes', scale_factor=self.size / 20)
        mlab.points3d(x, y, z, self.Gen.detected_luminosities ** (1 / 3), color=(1.0, 1.0, 1.0), mode='sphere',
                      opacity=1, scale_factor=1)
        # mlab.points3d(0, 0, 0, 1, color=(0.0, 0.0, 1.0), mode='sphere', scale_factor = 1)
        mlab.points3d(0, 0, 0, color=(1.0, 1.0, 1.0), mode='axes', scale_factor=self.Gen.size / 20)
        for (xhat, yhat, zhat, s) in zip(*zip(*self.Gen.BH_true_coords), self.Gen.BH_true_luminosities ** (1 / 3)):
            mlab.points3d(xhat, yhat, zhat, s * 1.01, color=(0.0, 1.0, 0.0), mode='sphere', opacity=1,
                          scale_factor=1)
        for (xhat, yhat, zhat, s) in zip(*zip(*self.Gen.BH_detected_coords), self.Gen.BH_detected_luminosities ** (1 / 3)):
            mlab.points3d(xhat, yhat, zhat, s, color=(1.0, 0.0, 0.0), mode='sphere', opacity=1, scale_factor=1)
        mlab.points3d(0, 0, 0, self.Gen.max_D * 2, color=(1.0, 1.0, 1.0), mode='sphere', opacity=0.05, scale_factor=1)

        for i, PDF in enumerate(self.Gen.BH_detected_meshgrid):
            X, Y, Z = self.Gen.BH_contour_meshgrid
            n = 1000
            PDF = PDF / PDF.sum()
            t = np.linspace(0, PDF.max(), n)
            integral = (((PDF >= t[:, None, None, None]) * PDF).sum(axis=(1, 2, 3)))
            f = interpolate.interp1d(integral, t)
            t_contours = f(np.array([0.9973, 0.9545, 0.6827]))

            mlab.contour3d(X, Y, Z, PDF, contours=[*t_contours], opacity=0.15, colormap="RdBu")

        mlab.outline(extent=[-self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size, -self.Gen.size, self.Gen.size])
        mlab.show()
