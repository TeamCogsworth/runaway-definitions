import cogsworth
import astropy.units as u
import numpy as np

import sfh_model

p = cogsworth.pop.Population(10_000_000, final_kstar1=[13, 14], 
                             m1_cutoff=7, processes=48,
                             sfh_model=sfh_model.ClusteredRecentNearSun,
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