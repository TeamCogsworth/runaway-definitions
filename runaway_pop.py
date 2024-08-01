import cogsworth
import astropy.units as u
import numpy as np

class RecentNearSun(cogsworth.sfh.Wagg2022):
    def __init__(self, components=["low_alpha_disc"], component_masses=[1],
                 near_thresh=3 * u.kpc, **kwargs):
        self.near_thresh = near_thresh
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
            
        return np.random.uniform(8.122 - near_thresh.to(u.kpc).value,
                                 8.122 + near_thresh.to(u.kpc).value, size) * u.kpc

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        U = np.random.rand(size)
        norm = 1 / (self.tsfr * np.exp(-self.galaxy_age / self.tsfr) * (np.exp(200 * u.Myr / self.tsfr) - 1))
        tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr) + 1)

        return tau
    
    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        # create an array of which component each point belongs to
        self._which_comp = np.repeat("low_alpha_disc", self._size)

        self._tau = self.draw_lookback_times(self._size)
        sun_x, sun_y = self.sun_pos(-self._tau)
        
        angle_offset = np.random.uniform(0, 2 * np.pi, self._size) * u.rad
        r_offset = np.random.rand(self._size)**(0.5) * self.near_thresh
        x_offset, y_offset = r_offset * np.cos(angle_offset), r_offset * np.sin(angle_offset)
        x, y = sun_x + x_offset, sun_y + y_offset
        
        rho = ((x**2 + y**2)**(0.5)).to(u.kpc)
        z = self.draw_heights(self._size)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(self._size)
        self._tau = self._tau[random_order]
        rho = rho[random_order]
        z = z[random_order]
        self._which_comp = self._which_comp[random_order]

        self._x = x[random_order]
        self._y = y[random_order]
        self._z = z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()
    

p = cogsworth.pop.Population(10_000_000, final_kstar1=[13, 14], m1_cutoff=7, processes=48,
                             sfh_model=RecentNearSun, max_ev_time=200 * u.Myr,
                             store_entire_orbits=False)
p.create_population()
p.save("/mnt/home/twagg/ceph/pops/mw-recent-200-nearsun.h5", overwrite=True)