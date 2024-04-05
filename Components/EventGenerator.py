#%%

import numpy as np
import random
import scipy as sp
import scipy.special as sc
from matplotlib import pyplot as plt
from scipy import interpolate
from Visualising.Visualiser_3d import Visualiser_3d
from Tools import BB1_sampling as BB1Pack
import scipy.stats as ss
from Components.Universe import Universe
from Components.SurveyAndEventData import SurveyAndEventData


class EventGenerator(Universe):
    def __init__(self, dimension = 2, luminosity_gen_type = "Fixed",
                 coord_gen_type = "Clustered",
                 cluster_coeff = 2, total_luminosity = 1000, size = 1,
                 alpha = .3, beta=-1.5, characteristic_luminosity = 1, min_lum = 0,
                 max_lum = 1, lower_lim=1, event_rate = 1, sample_time = .01 ,event_distribution = "Random",
                 noise_distribution = "BVMF_eff", contour_type = "BVM",
                 noise_std = 3, resolution = 400, BVM_c = 15, H_0 = 70,
                 BVM_k = 2, BVM_kappa = 200, redshift_noise_sigma = 0,
                 centroid_n=6, centroid_sigma=0.1,
                 plot_contours = True, cube=True, seed = None, event_count_type = "Poisson"):

        super().__init__(dimension = dimension, luminosity_gen_type = luminosity_gen_type,
                         coord_gen_type = coord_gen_type,
                         cluster_coeff = cluster_coeff, total_luminosity = total_luminosity,
                         size = size, alpha = alpha, beta = beta, characteristic_luminosity = characteristic_luminosity,
                         min_lum = min_lum, max_lum = max_lum, lower_lim = lower_lim, redshift_noise_sigma = redshift_noise_sigma,
                         seed = seed, H_0 = H_0, cube = cube,
                         centroid_n=centroid_n, centroid_sigma=centroid_sigma,
                         )

        self.BVM_k = BVM_k
        self.BVM_c = BVM_c
        self.BVM_kappa = BVM_kappa

        self.plot_contours = plot_contours

        self.event_distribution = event_distribution
        
        self.event_rate = event_rate
        self.sample_time = sample_time
        self.event_count_type = event_count_type
        self.event_count_generator =  dict({
            "Poisson" : self.poisson_event_count,
        })
        self.event_count = self.event_count_generator[self.event_count_type]()

        self.resolution = resolution
        self.confidence_levels = [0.9973,  0.9545, 0.6827]

        self.event_choice = event_distribution

        self.noise_distribution = noise_distribution

        self.noise_sigma = self.gauss_std_from_burr(noise_std*self.size/20)

        self.BH_galaxies = np.empty((self.event_count), dtype = object)
        self.BH_true_coords = np.zeros((self.event_count, self.dimension))
        self.BH_detected_luminosities = np.zeros((self.event_count))
        self.BH_true_luminosities = np.zeros((self.event_count))
        self.BH_detected_coords = np.empty((self.event_count, self.dimension))

        self.contour_type = contour_type +"_"+str(self.dimension)+"d"

        if self.plot_contours:
            if self.dimension == 2:
                self.BH_contour_meshgrid = np.meshgrid(np.linspace(-self.size, self.size, self.resolution),
                                                    np.linspace(-self.size, self.size, self.resolution))
            if self.dimension == 3:
                self.BH_contour_meshgrid = np.mgrid[-self.size:self.size:complex(0,self.resolution),
                                        -self.size:self.size:complex(0,self.resolution),
                                        -self.size:self.size:complex(0,self.resolution)]

            self.BH_detected_meshgrid = np.empty((self.event_count, *np.shape(self.BH_contour_meshgrid[0])))



        self.event_generator = dict({"Random": self.random_galaxy,
                                     "Proportional":self.proportional_galaxy})
        self.coord_noise_generator = dict(
                                           {"gauss": self.gauss_noise,
                                            "BVM": self.BVM_sample,
                                            "BVMF_eff": self.BVMF_sample_eff,
                                            "BVMF_r2_eff": self.BVMF_r2_sample_eff,
                                           "GVMF_eff": self.GVMF_sample_eff,
                                            "GJVMF_eff": self.GJVMF_sample_eff
                                            })
        self.contour_generator = dict({"gauss_2d": self.gauss_2d,
                                       "gauss_3d": self.gauss_3d,
                                       "GVMF_2d": self.GVMF_eff_contour,
                                       "GVMF_3d": self.GVMF_eff_contour,
                                       "BVM_2d": self.BVMShell,
                                       "BVM_3d" : self.BVMShell_3d
                                       })
        self.generate_events(event_distribution = self.event_distribution,
                             noise_distribution = self.noise_distribution)

    def generate_events(self, event_distribution, noise_distribution):
        event_count = 0
        detected_event_count = 0
        detected_event_indices = []
        while event_count < self.event_count:
            selected = self.event_generator[event_distribution]()
            mu = self.true_coords[selected]
            noise = self.coord_noise_generator[noise_distribution](mu)
            if np.sqrt(np.sum(np.square(self.true_coords[selected] + noise))) > self.max_D:
                event_count += 1
                continue

            self.BH_galaxies[event_count] = self.galaxies[selected]
            self.BH_true_coords[event_count] = self.true_coords[selected]
            self.BH_detected_coords[event_count] = self.true_coords[selected] + noise
            self.BH_true_luminosities[event_count] = self.true_luminosities[selected]
            self.BH_detected_luminosities[event_count] = self.detected_luminosities[selected]
            if self.plot_contours:
                grid = self.contour_generator[self.contour_type](self.BH_detected_coords[event_count])
                self.BH_detected_meshgrid[event_count] = grid/grid.sum()
            detected_event_indices.append(event_count)
            event_count+=1
            detected_event_count += + 1

        self.detected_event_count = detected_event_count

        self.BH_galaxies = self.BH_galaxies[detected_event_indices]
        self.BH_true_coords = self.BH_true_coords[detected_event_indices,:]
        self.BH_detected_luminosities = self.BH_detected_luminosities[detected_event_indices]
        self.BH_true_luminosities = self.BH_true_luminosities[detected_event_indices]
        self.BH_detected_coords = self.BH_detected_coords[detected_event_indices,:]
        if self.plot_contours:
            self.BH_detected_meshgrid = self.BH_detected_meshgrid[detected_event_indices,:]

    def gauss_std_from_burr(self, l):
        s = np.sqrt(-self.burr_u(1,l)**2 + self.burr_u(2,l))
        return s
    
    def burr_u(self, n, l):
        return (l**n)*self.BVM_k*sc.beta(self.BVM_k - n/self.BVM_c , 1 + n/self.BVM_c)

    def poisson_event_count(self):
        # self.np_rand_state2 = np.random.default_rng(1)
        self.total_event_count = self.np_rand_state.poisson(lam = self.event_rate*np.sum(self.true_luminosities)*self.sample_time)
        return self.total_event_count

    def random_galaxy(self):
        return self.rand_rand_state.randint(0, self.n-1)

    def proportional_galaxy(self):
        n_list = list(np.arange(0,self.n))
        source = self.rand_rand_state.choices(n_list, weights=self.true_luminosities)[0]
        return source

    def gauss_noise(self, mu):
        # lamb = np.linalg.norm(mu)
        #noise = self.np_rand_state.normal(loc=0.0, scale=self.gauss_std_from_burr(lamb), size=self.dimension)
        noise = self.np_rand_state.normal(loc=0.0, scale=self.noise_sigma, size=self.dimension)
        return noise

    def BVM_sample(self, mu):
        if self.dimension == 3:
            r_grid = np.linspace(0, np.sqrt(3)*self.size, 10*self.resolution)
            b_r = np.linalg.norm(mu)
            burr_w = (r_grid**2)*self.burr(r_grid, self.BVM_c, self.BVM_k, b_r)
            burr_w = burr_w/np.sum(burr_w)
            self.dbug1 = burr_w
            r_samp = self.np_rand_state.choice(r_grid, p=burr_w)
            
            phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 10*self.resolution),
                            np.linspace(0, np.pi, 10*self.resolution))
            phi_mu = np.arctan2(mu[1], mu[0])
            theta_mu = np.arctan2(np.sqrt(mu[0]**2 + mu[1]**2), mu[2])

            vm_weight = 4*np.pi*self.von_misses_fisher_3d(phi, theta, phi_mu, theta_mu, self.BVM_kappa)
            vm_weight = vm_weight/np.sum(vm_weight)
            flat = vm_weight.flatten()
            sample_index = self.np_rand_state.choice(len(flat), p=flat)
            # aaaaaaa = self.np_rand_state.choice(len(flat), p=flat)
            adjusted_index = np.unravel_index(sample_index, vm_weight.shape)
            phi_samp = phi[adjusted_index]
            theta_samp = theta[adjusted_index]

            x = r_samp*np.sin(theta_samp)*np.cos(phi_samp)
            y = r_samp*np.sin(theta_samp)*np.sin(phi_samp)
            z = r_samp*np.cos(theta_samp)
            sample = np.array([x,y,z])

            return sample - mu

        elif self.dimension == 2:
            r_grid = np.linspace(0, np.sqrt(2)*self.size, 1000*self.resolution)
            b_r = np.linalg.norm(mu)

            burr_w = r_grid*self.burr(r_grid, self.BVM_c, self.BVM_k, b_r)
            burr_w = burr_w/np.sum(burr_w)
            r_samp = self.np_rand_state.choice(r_grid, p=burr_w)

            phi_grid = np.linspace(0, 2*np.pi, 1000*self.resolution)
            phi_mu = np.arctan2(mu[1], mu[0])

            vm_weight = self.von_misses(phi_grid, phi_mu, self.BVM_kappa)
            vm_weight = vm_weight/np.sum(vm_weight)
            phi_samp = self.np_rand_state.choice(phi_grid, p=vm_weight)

            x = r_samp*np.cos(phi_samp)
            y = r_samp*np.sin(phi_samp)
            sample = np.array([x,y])

            return sample - mu
        
    def BVMF_sample_eff(self, mu):
        lamb = np.linalg.norm(mu)
        r_samp = BB1Pack.burr_sampling(self.np_rand_state, self.BVM_c, self.BVM_k, lamb, 1)[0]
        if self.dimension == 2:
            phi_mu = np.arctan2(mu[1], mu[0])
            phi_samp = BB1Pack.von_misses_sampling(self.rand_rand_state, phi_mu, self.BVM_kappa, 1)[0]
            x = r_samp * np.cos(phi_samp)
            y = r_samp * np.sin(phi_samp)
            sample = np.array([x, y])
            return sample - mu
        elif self.dimension == 3:
            norm_dir_samp = BB1Pack.von_misses_fisher_sampling(self.np_rand_state, mu, self.BVM_kappa, 1)[0]
            sample = r_samp * norm_dir_samp
            return sample - mu

    def GVMF_sample_eff(self, mu):
        lamb = np.linalg.norm(mu)
        r_samp = self.np_rand_state.normal(loc=lamb, scale=self.noise_sigma)
        if self.dimension == 2:
            phi_mu = np.arctan2(mu[1], mu[0])
            phi_samp = BB1Pack.von_misses_sampling(self.rand_rand_state, phi_mu, self.BVM_kappa, 1)[0]
            x = r_samp * np.cos(phi_samp)
            y = r_samp * np.sin(phi_samp)
            sample = np.array([x, y])
            return sample - mu
        elif self.dimension == 3:
            norm_dir_samp = BB1Pack.von_misses_fisher_sampling(self.np_rand_state, mu, self.BVM_kappa, 1)[0]
            sample = r_samp * norm_dir_samp
            return sample - mu

    def GJVMF_sample_eff(self, mu):
        lamb = np.linalg.norm(mu)
        r_samp = self.polynorm_sample(lamb)
        if self.dimension == 2:
            phi_mu = np.arctan2(mu[1], mu[0])
            phi_samp = BB1Pack.von_misses_sampling(self.rand_rand_state, phi_mu, self.BVM_kappa, 1)[0]
            x = r_samp * np.cos(phi_samp)
            y = r_samp * np.sin(phi_samp)
            sample = np.array([x, y])
            return sample - mu
        elif self.dimension == 3:
            norm_dir_samp = BB1Pack.von_misses_fisher_sampling(self.np_rand_state, mu, self.BVM_kappa, 1)[0]
            sample = r_samp * norm_dir_samp
            return sample - mu
        
    def BVMF_r2_sample_eff(self, mu):
        lamb = np.linalg.norm(mu)
        r_samp = BB1Pack.r2_burr_sampling(self.rand_rand_state, self.BVM_c, self.BVM_k, lamb, 1)[0]
        if self.dimension == 2:
            phi_mu = np.arctan2(mu[1], mu[0])
            phi_samp = BB1Pack.von_misses_sampling(self.rand_rand_state, phi_mu, self.BVM_kappa, 1)[0]
            x = r_samp*np.cos(phi_samp)
            y = r_samp*np.sin(phi_samp)
            sample = np.array([x,y])
            return sample - mu
        elif self.dimension == 3:
            norm_dir_samp = BB1Pack.von_misses_fisher_sampling(self.np_rand_state, mu, self.BVM_kappa, 1)[0]
            sample = r_samp*norm_dir_samp
            return sample - mu

    def polynorm_sample(self, lamb):
        N = 25
        a = 1
        b = -lamb
        c = -2*self.noise_sigma
        center = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        envelope_max = np.exp(-0.5*((center - lamb)/self.noise_sigma)**2)*center**2
        xs = self.np_rand_state.uniform(low = center - 5*self.noise_sigma, high = center + 5*self.noise_sigma, size = N)
        ys = self.np_rand_state.uniform(low = 0, high = envelope_max, size = N)
        ys_limit = np.exp(-0.5 * ((xs - lamb) / self.noise_sigma) ** 2) * xs ** 2
        xs_accepted = xs[np.where(ys <= ys_limit)]
        if len(xs_accepted)==0:
            return self.polynorm_sample(lamb)
        return xs_accepted[0]


    def plot_universe_and_events(self, show = True):
        if self.dimension == 2:
            fig, ax = self.plot_universe(show = False)
            if self.detected_event_count != 0:
                x, y = zip(*self.BH_true_coords)
                for (xhat, yhat, s) in zip(*zip(*self.BH_detected_coords), self.BH_detected_luminosities):
                    ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=self.L_star, color="red", zorder = 4))
                if self.plot_contours is True:
                    for i, Z in enumerate(self.BH_detected_meshgrid):
                        X, Y = self.BH_contour_meshgrid
                        z = Z
                        n = 1000
                        z = z / z.sum()
                        t = np.linspace(0, z.max(), n)
                        integral = ((z >= t[:, None, None]) * z).sum(axis=(1, 2))
                        f = interpolate.interp1d(integral, t)
                        t_contours = f(np.array([0.9973, 0.9545, 0.6827]))
                        ax.contour(X,Y, z, t_contours, colors="red", zorder = 2)
                for (x, y, s) in zip(x, y, self.BH_true_luminosities):
                    ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="g", zorder = 4))
            if show:
                plt.show()
            else: return fig, ax
        else:
            return Visualiser_3d(Gen = self).plot_universe_and_events(show = show)


    def gauss_2d(self, mu):
        sig_x = self.noise_sigma**2
        sig_y = self.noise_sigma**2
        sig_xy = 0
        x = self.BH_contour_meshgrid[0]
        y = self.BH_contour_meshgrid[1]
        rv = sp.stats.multivariate_normal(mu, [[sig_x, sig_xy], [sig_xy, sig_y]])
        Z = rv.pdf(np.dstack((x, y)))
        return Z

    def gauss_3d(self, mu):
        sig_x = self.noise_sigma**2
        sig_y = self.noise_sigma**2
        sig_z = self.noise_sigma**2
        sig_xy = 0
        sig_xz = 0
        sig_yz = 0
        x = self.BH_contour_meshgrid[0]
        y = self.BH_contour_meshgrid[1]
        z = self.BH_contour_meshgrid[2]
        rv = sp.stats.multivariate_normal(mu, [[sig_x, sig_xy, sig_xz], [sig_xy, sig_y, sig_yz], [sig_xz, sig_yz, sig_z]])
        Z = rv.pdf(np.dstack((x, y, z)))
        return Z

    def d2_gauss(self, X, Y, u_x, u_y, s_x, s_y):
        Z = np.exp(-(((X-u_x)/s_x)**2 + ((Y-u_y)/s_y)**2)/2)/(2*np.pi*s_x*s_y)
        return Z

    def von_misses(self, x, u, kappa):
        return np.exp(kappa*np.cos(x-u))/(2*np.pi*sp.special.iv(0,kappa))

    def burr(self, x, c, k, l):
        return (c*k/l)*((x/l)**(c-1))*((1+(x/l)**c)**(-k-1))

    def von_misses_fisher_3d(self, phi, theta, u_phi, u_theta, kappa):
        # normalisation specific to 3-sphere
        # x, u are vectors
        # ||u|| = 1
        # kappa >= 0
        C = kappa/(2*np.pi*(np.exp(kappa) - np.exp(-kappa)))
        return C * np.exp(kappa*(np.sin(theta)*np.sin(u_theta)*np.cos(phi-u_phi) + np.cos(theta)*np.cos(u_theta)))

    def BVMShell(self,mu):
        u_x = mu[0]
        u_y = mu[1]
        s_x = self.noise_sigma
        s_y = self.noise_sigma
        X = self.BH_contour_meshgrid[0]
        Y = self.BH_contour_meshgrid[1]
        r = np.sqrt((X) ** 2 + (Y) ** 2)
        phi = np.arctan2(Y, X)
        u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2)
        u_phi = np.arctan2(u_y, u_x)

        k = self.BVM_k
        c = self.BVM_c
        kappa = self.BVM_kappa

        angular = self.von_misses(phi, u_phi, kappa)
        radial = self.burr(r, c, k, u_r)
        Z = r * angular * radial

        # vals = [self.d2_gauss(u_x + 3 * s_x, u_y + 3 * s_y, u_x, u_y, s_x, s_y),
        #         self.d2_gauss(u_x + 2 * s_x, u_y + 2 * s_y, u_x, u_y, s_x, s_y),
        #         self.d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
        return Z

    def BVMShell_3d(self,mu):
        u_x = mu[0]
        u_y = mu[1]
        u_z = mu[2]
        # s_x = self.noise_sigma
        # s_y = self.noise_sigma
        # s_z = self.noise_sigma
        X = self.BH_contour_meshgrid[0]
        Y = self.BH_contour_meshgrid[1]
        Z = self.BH_contour_meshgrid[2]
        r = np.sqrt((X) ** 2 + (Y) ** 2 + (Z) ** 2)
        u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2 + (u_z) ** 2)

        phi = np.arctan2(Y, X)
        u_phi = np.arctan2(u_y, u_x)
        XY = np.sqrt((X) ** 2 + (Y) ** 2)
        theta = np.arctan2(XY, Z)
        u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)

        k = self.BVM_k
        c = self.BVM_c
        kappa = self.BVM_kappa
        # u_ang = mu/np.linalg.norm(mu)

        # angular = self.von_misses_fisher_3d(x, u_ang, kappa)
        angular = self.von_misses_fisher_3d(phi, theta, u_phi, u_theta, kappa)
        radial = self.burr(r, c, k, u_r)
        f = (np.sin(theta) * r ** 2) * angular * radial

        # vals = [self.d2_gauss(u_x + 3 * s_x, u_y + 3 * s_y, u_x, u_y, s_x, s_y),
        #        self.d2_gauss(u_x + 2 * s_x, u_y + 2 * s_y, u_x, u_y, s_x, s_y),
        #        self.d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
        return f


    def GVMF_eff_contour(self,mu):
        if self.dimension ==3:
            u_x = mu[0]
            u_y = mu[1]
            u_z = mu[2]
            # s_x = self.noise_sigma
            # s_y = self.noise_sigma
            # s_z = self.noise_sigma
            X = self.BH_contour_meshgrid[0]
            Y = self.BH_contour_meshgrid[1]
            Z = self.BH_contour_meshgrid[2]
            r = np.sqrt((X) ** 2 + (Y) ** 2 + (Z) ** 2)
            u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2 + (u_z) ** 2)

            phi = np.arctan2(Y, X)
            u_phi = np.arctan2(u_y, u_x)
            XY = np.sqrt((X) ** 2 + (Y) ** 2)
            theta = np.arctan2(XY, Z)
            u_theta = np.arctan2(np.sqrt(u_x ** 2 + u_y ** 2), u_z)

            k = self.BVM_k
            c = self.BVM_c
            kappa = self.BVM_kappa
            # u_ang = mu/np.linalg.norm(mu)

            # angular = self.von_misses_fisher_3d(x, u_ang, kappa)
            angular = self.von_misses_fisher_3d(phi, theta, u_phi, u_theta, kappa)
            radial = ss.norm.pdf(x = r, loc = u_r, scale = self.noise_sigma)
            f = (np.sin(theta) * r ** 2) * angular * radial

            # vals = [self.d2_gauss(u_x + 3 * s_x, u_y + 3 * s_y, u_x, u_y, s_x, s_y),
            #        self.d2_gauss(u_x + 2 * s_x, u_y + 2 * s_y, u_x, u_y, s_x, s_y),
            #        self.d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
            return f
        if self.dimension == 2:
            u_x = mu[0]
            u_y = mu[1]
            s_x = self.noise_sigma
            s_y = self.noise_sigma
            X = self.BH_contour_meshgrid[0]
            Y = self.BH_contour_meshgrid[1]
            r = np.sqrt((X) ** 2 + (Y) ** 2)
            phi = np.arctan2(Y, X)
            u_r = np.sqrt((u_x) ** 2 + (u_y) ** 2)
            u_phi = np.arctan2(u_y, u_x)

            k = self.BVM_k
            c = self.BVM_c
            kappa = self.BVM_kappa

            angular = self.von_misses(phi, u_phi, kappa)
            radial = ss.norm.pdf(x = r, loc = u_r, scale = self.noise_sigma)
            Z = r * angular * radial

            # vals = [self.d2_gauss(u_x + 3 * s_x, u_y + 3 * s_y, u_x, u_y, s_x, s_y),
            #         self.d2_gauss(u_x + 2 * s_x, u_y + 2 * s_y, u_x, u_y, s_x, s_y),
            #         self.d2_gauss(u_x + s_x, u_y + s_y, u_x, u_y, s_x, s_y)]
            return Z

    def GetSurveyAndEventData(self, min_flux = 0, survey_incompleteness = 0, completeness_type = 'cut_lim', DD = 0):
        return SurveyAndEventData(dimension = self.dimension, detected_coords = self.detected_coords,
                                  detected_luminosities = self.detected_luminosities,
                                  fluxes = self.fluxes, BH_detected_coords = self.BH_detected_coords, BVM_k = self.BVM_k,
                                  BVM_c = self.BVM_c, BVM_kappa = self.BVM_kappa, BurrFunc = self.burr, VonMissesFunc= self.von_misses, 
                                  VonMissesFisherFunc = self.von_misses_fisher_3d, detected_redshifts=self.detected_redshifts,
                                  detected_redshifts_uncertainties = self.detected_redshifts_uncertainties, contour_type=self.contour_type,
                                  max_D = self.max_D, d_ratio = self.d_ratio, detected_event_count=self.detected_event_count,
                                  sample_time = self.sample_time, noise_distribution = self.noise_distribution, noise_sigma = self.noise_sigma, redshift_noise_sigma = self.redshift_noise_sigma,
                                  min_flux = min_flux, survey_incompleteness = survey_incompleteness, completeness_type = completeness_type, DD = DD, event_rate = self.event_rate, c = self.c, event_distribution = self.event_distribution,
                                  alpha = self.alpha, beta = self.beta, characteristic_luminosity = self.L_star, min_lum = self.lower_lim)

#
#
#
# Gen = EventGenerator(dimension = 3, size = 50, resolution = 100,
#                       luminosity_gen_type = "Cut-Schechter", coord_gen_type = "Random",
#                       cluster_coeff=5, characteristic_luminosity=1, total_luminosity=10/3, sample_time=0.09, event_rate=10,
#                       event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0.0, plot_contours=True, seed = 10)
# # print("plotting")
# # print(Gen.detected_event_count)
# Gen.plot_universe_and_events()




# %%
