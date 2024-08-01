import cogsworth
import astropy.units as u
import numpy as np

class ClusteredRecentNearSun(cogsworth.sfh.Wagg2022):
    def __init__(self, components=["low_alpha_disc"], component_masses=[1],
                 near_thresh=3 * u.kpc,
                 cluster_radius=1 * u.pc,
                 n_per_cluster=10000, **kwargs):
        self.near_thresh = near_thresh
        self.cluster_radius = cluster_radius
        self.n_per_cluster = n_per_cluster
        super().__init__(components=components, component_masses=component_masses, **kwargs)

    def sun_pos(self, t):
        sun_R = 8.122 * u.kpc
        sun_v = 229.40403 * u.km / u.s
        T = ((2 * np.pi * sun_R) / sun_v).to(u.Myr)
        theta = ((2 * np.pi * t / T) % (2 * np.pi)).decompose() * u.rad
        x = sun_R * np.cos(theta)
        y = sun_R * np.sin(theta)
        return x, y

    def draw_radii(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        return np.random.uniform(8.122 - self.near_thresh.to(u.kpc).value,
                                 8.122 + self.near_thresh.to(u.kpc).value, size) * u.kpc

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        U = np.random.rand(size)
        norm = 1 / (self.tsfr * np.exp(-self.galaxy_age / self.tsfr) * (np.exp(200 * u.Myr / self.tsfr) - 1))
        tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr) + 1)

        return tau

    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        n_clusters = int(np.ceil(self._size / self.n_per_cluster))
        # create an array of which component each point belongs to
        self._which_comp = np.repeat("low_alpha_disc", self._size)

        self._tau = self.draw_lookback_times(n_clusters)
        sun_x, sun_y = self.sun_pos(-self._tau)

        angle_offset = np.random.uniform(0, 2 * np.pi, n_clusters) * u.rad
        r_offset = np.random.rand(n_clusters)**(0.5) * self.near_thresh
        x_offset, y_offset = r_offset * np.cos(angle_offset), r_offset * np.sin(angle_offset)
        x, y = sun_x + x_offset, sun_y + y_offset

        rho = ((x**2 + y**2)**(0.5)).to(u.kpc)
        z = self.draw_heights(n_clusters)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(n_clusters)
        self._tau = self._tau[random_order]
        rho = rho[random_order]
        z = z[random_order]
        x = x[random_order]
        y = y[random_order]

        self._tau = np.repeat(self._tau, self.n_per_cluster)[:self._size]
        rho = np.repeat(rho, self.n_per_cluster)[:self._size]
        x = np.repeat(x, self.n_per_cluster)[:self._size]
        y = np.repeat(y, self.n_per_cluster)[:self._size]
        z = np.repeat(z, self.n_per_cluster)[:self._size]

        # spread out each cluster
        x, y, z = np.random.normal([x.to(u.kpc).value, y.to(u.kpc).value, z.to(u.kpc).value],
                                   self.cluster_radius.to(u.kpc).value / np.sqrt(3),
                                   size=(3, self._size)) * u.kpc

        self._x = x
        self._y = y
        self._z = z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()
    

p = cogsworth.pop.Population(10_000_000, final_kstar1=[13, 14], 
                             m1_cutoff=7, processes=48,
                             sfh_model=ClusteredRecentNearSun,
                             sfh_params={
                                "near_thresh": 3 * u.kpc,
                                "cluster_radius": 1 * u.pc,
                                "n_per_cluster": 10000,
                             },
                             velocity_dispersion=5 * u.km / u.s,
                             max_ev_time=200 * u.Myr,
                             store_entire_orbits=False)
p.create_population()
p.save("/mnt/home/twagg/ceph/pops/mw-recent-200-nearsun-clustered.h5", overwrite=True)