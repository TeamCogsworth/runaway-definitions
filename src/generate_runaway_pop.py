import cogsworth
import astropy.units as u
import numpy as np
import argparse

import sfh_model

def main():
    parser = argparse.ArgumentParser(description="Generate a population of runaway stars")
    parser.add_argument("-o", "--output-file", type=str, help="Output file to save the population to")
    parser.add_argument('-d', '--dispersion', type=float, help="Velocity dispersion of the runaway stars")
    parser.add_argument('-r', '--radius', type=float, help="Cluster radius")

    args = parser.parse_args()

    p = cogsworth.pop.Population(10_000_000, final_kstar1=[13, 14], 
                                m1_cutoff=7, processes=48,
                                sfh_model=sfh_model.ClusteredRecentNearSun,
                                sfh_params={
                                    "near_thresh": 3 * u.kpc,
                                    "cluster_radius": args.radius * u.pc,
                                    "n_per_cluster": 10000,
                                },
                                v_dispersion=args.dispersion * u.km / u.s,
                                max_ev_time=200 * u.Myr,
                                store_entire_orbits=False)
    p.create_population()
    p.save(args.output_file, overwrite=True)

if __name__ == "__main__":
    main()