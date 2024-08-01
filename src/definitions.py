import numpy as np
import astropy.units as u

def get_theorist_runaways(p, teff_thresh=30000, vel_thresh=30, near_thresh=3 * u.kpc):
    """Identify runaway stars in a population of binaries based on theorist criteria
    
    For theorists, we consider the system immediately after the first supernova.
    Either of the following two cases need to be met:
        - System is bound after SN (`sep > 0`), companion has `teff > 30kK`, `vsys_1_total > 30 km/s`
        - Or system is disrupted after SN (`sep < 0`), ejected companion has `teff > 30kK`, `vsys_2_total > 30 km/s`
    
    In both cases the system must be nearby to the sun (within 3 kpc by default).

    Parameters
    ----------
    p : :class:`cogsworth.pop.Population`
        A population of binaries
    teff_thresh : `int`, optional
        Temperature threshold in Kelvin, by default 30000
    vel_thresh : `int`, optional
        Runaway velocity threshold in km/s, by default 30
    near_thresh : `astropy.units.Quantity` [length], optional
        Threshold for whether a system is nearby, by default 3*u.kpc

    Returns
    -------
    theorist_runaway_pop : :class:`cogsworth.pop.Population`
        A population of runaway stars
    theorist_masks : `dict`
        A dictionary of masks used to identify runaway stars
    """
    # get the first kick information for each system (only for systems that received a kick)
    first_kicks = p.kick_info.drop_duplicates(subset="bin_num", keep="first")
    first_kicks = first_kicks[first_kicks["star"] != 0]
    
    # get the evolution information for the system immediately after the first supernova
    first_kick_bpp = p.bpp[p.bpp["evol_type"].isin([15, 16])].drop_duplicates(subset="bin_num", keep="first")
    after_sn_1 = p.bpp.iloc[first_kick_bpp["row_ind"] + 1]

    # define a series of masks to identify runaway stars for the primary and secondary
    theorist_masks = {
        "co": {i: after_sn_1[f"kstar_{i}"].isin([13, 14]) for i in [1, 2]},
        "ms": {i: after_sn_1[f"kstar_{i}"] <= 1 for i in [1, 2]},
        "hot": {i: after_sn_1[f"teff_{i}"] >= teff_thresh for i in [1, 2]},
        "fast": {i: first_kicks[f"vsys_{i}_total"] >= vel_thresh for i in [1, 2]},
        "bound": after_sn_1["sep"] > 0.0,
        "disrupted": after_sn_1["sep"] < 0.0,
        "nearby": {}
    }

    # mask for systems that are nearby to the sun
    sun_loc = np.array([8.122, 0, 0]) * u.kpc
    dist_mask = np.linalg.norm(p.final_pos - sun_loc, axis=1) < near_thresh

    # bound systems are the same as primaries so we just need those
    bound_nearby = dist_mask[:len(p)]

    # for disrupted systems we can overwrite the nearby mask for the disrupted secondary
    dis_nearby = dist_mask[:len(p)]
    dis_nearby[p.disrupted] = dist_mask[len(p):]

    # then just save it for systems that had a supernova
    had_sn = np.isin(p.bin_nums, first_kick_bpp["bin_num"])
    theorist_masks["nearby"][1] = bound_nearby[had_sn]
    theorist_masks["nearby"][2] = dis_nearby[had_sn]
    
    # bound runaway stars are where the system is fast, nearby and contains a compact object + a hot MS star
    co_plus_o = ((theorist_masks["co"][1] & theorist_masks["ms"][2] & theorist_masks["hot"][2])
                 | (theorist_masks["co"][2] & theorist_masks["ms"][1] & theorist_masks["hot"][1]))
    theorist_bound_runaways = (theorist_masks["bound"]
                               & theorist_masks["fast"][1]
                               & theorist_masks["nearby"][1]
                               & co_plus_o)
    
    # for disrupted systems the primary will be a compact object so let's just focus on the secondary
    # this star must be fast, nearby and hot (i.e. an O star)
    theorist_disrupted_runaways = (theorist_masks["disrupted"]
                                   & theorist_masks["fast"][2]
                                   & theorist_masks["ms"][2]
                                   & theorist_masks["hot"][2]
                                   & theorist_masks["nearby"][2])

    # mask out the population
    theorist_runaway_mask = theorist_bound_runaways | theorist_disrupted_runaways
    theorist_runaway_pop = p[first_kicks["bin_num"].values[theorist_runaway_mask].astype(int)]
    
    return theorist_runaway_pop, theorist_masks


def get_observer_population(p, teff_thresh=30000, vel_thresh=30, near_thresh=3 * u.kpc, v_circ=None):
    """Identify runaway stars in a population of binaries based on observer criteria

    Parameters
    ----------
    p : :class:`cogsworth.pop.Population`
        A population of binaries
    teff_thresh : `int`, optional
        Temperature threshold in Kelvin, by default 30000
    vel_thresh : `int`, optional
        Runaway velocity threshold in km/s, by default 30
    near_thresh : `astropy.units.Quantity` [length], optional
        Threshold for whether a system is nearby, by default 3*u.kpc
    v_circ : :class:`astropy.units.Quantity` [velocity], optional
        Circular velocity at the location of each system at present day, by default None

    Returns
    -------
    observer_runaway_pop : :class:`cogsworth.pop.Population`
        A population of runaway stars
    observer_masks : dict
        A dictionary of masks used to identify runaway stars
    """
    # get the circular velocity at the location of each system if not provided
    if v_circ is None:
        v_circ = p.galactic_potential.circular_velocity(p.final_pos.T)

    # calculate the masks for each observer criterion
    observer_masks = {
        "co": {i: p.final_bpp[f"kstar_{i}"].isin([13, 14]) for i in [1, 2]},
        "ms": {i: p.final_bpp[f"kstar_{i}"] <= 1 for i in [1, 2]},
        "hot": {i: p.final_bpp[f"teff_{i}"] >= teff_thresh for i in [1, 2]},
        "nearby": {}
    }

    # add some shortcuts for the o stars and bound systems
    observer_masks["o_star"] = {i: observer_masks["ms"][i] & observer_masks["hot"][i] for i in [1, 2]}
    bound = p.final_bpp["sep"] > 0.0
    
    # calculate nearby systems in the same way as ``get_theorist_runaways``
    sun_loc = np.array([8.122, 0, 0]) * u.kpc
    dist_mask = np.linalg.norm(p.final_pos - sun_loc, axis=1) < near_thresh
    bound_nearby = dist_mask[:len(p)]
    dis_nearby = dist_mask[:len(p)]
    dis_nearby[p.disrupted] = dist_mask[len(p):]

    observer_masks["nearby"][1] = bound_nearby
    observer_masks["nearby"][2] = dis_nearby
    
    # potential runaways can come in all shapes and sizes (these are masks before considering velocity)
    # disrupted systems can have runaway stars from secondary if that star is an o star
    # ignore primary because it will almost always be a compact object unless rarely secondary exploded first
    pr_dis_2 = p.disrupted & observer_masks["o_star"][2]

    # bound systems can be runaways if there is a compact object and an o star
    pr_bound = bound & ((observer_masks["co"][1] & observer_masks["o_star"][2])
                      | (observer_masks["co"][2] & observer_masks["o_star"][1]))

    # mergers can be runaways if there if it is an o star
    pr_mergers = (p.final_bpp["sep"] == 0.0) & (observer_masks["o_star"][1] | observer_masks["o_star"][2])
    
    # transform the circular velocity (v_circ) into cartesian coordinates
    final_phi = np.arctan2(p.final_pos[:, 1], p.final_pos[:, 0])
    final_rho = np.sum(p.final_pos[:, :2]**2, axis=1)**(0.5)
    v_phi = v_circ / final_rho
    v_circ_x = -final_rho * np.sin(final_phi) * v_phi
    v_circ_y = final_rho * np.cos(final_phi) * v_phi

    # calculate the relative velocity of each system to the circular velocity
    rel_vel = np.linalg.norm([p.final_vel[:, 0] - v_circ_x,
                              p.final_vel[:, 1] - v_circ_y,
                              p.final_vel[:, 2]], axis=0)
    fast = rel_vel > vel_thresh
    
    # concatenate all of the candidates that are moving fast enough
    observer_runaway_nums = np.concatenate((np.intersect1d(p.bin_nums[p.disrupted][fast[len(p):]],      # secondary disrupted and fast
                                                           p.bin_nums[pr_dis_2 & dis_nearby]),  
                                            p.bin_nums[(fast[:len(p)] & pr_mergers & bound_nearby)],    # mergers and fast
                                            p.bin_nums[(fast[:len(p)] & pr_bound & bound_nearby)]))     # bound and fast
    # de-duplicate the list of runaway stars just in case (shouldn't happen)
    observer_runaway_nums = np.unique(observer_runaway_nums)
    
    # mask out the population and return
    return p[observer_runaway_nums], observer_masks
