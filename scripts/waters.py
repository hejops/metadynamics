#!/usr/bin/env python3

"""
https://github.com/dkoes/md-scripts/blob/master/waters.py

Calculates the average number of water contacts for each residue across the
trajectory.  For each residue the waters are selected in a specified cutoff
distance and then the counts are averaged for that residue.

Output:
a .csv file with indexed by resid including residue names and the average water contacts
e.g.
resid | resname | waters
    1       LEU |    2.6 

Running the script
python waters.py MDAnalysis_supported_topolgy MDAnalysis_supported_trajectory csv_output_name.csv
e.g.
python contacts.py FXN_R165N_complex.prmtop FXN_R165N_complex1.dcd waters1.csv
"""

import MDAnalysis
import multiprocessing
import numpy as np  # numpy 1.16.6
import pandas as pd
import sys

from MDAnalysis.analysis import distances
from pandas import DataFrame


def read_traj(top: str, xtc: str):
    global xtc
    global curr_traj  # must be global to work with multiprocessing
    print("Loading trajectories...")
    return MDAnalysis.Universe(top, xtc)
    # curr_traj = traj
    # return traj


def get_res_waters(traj, cutoff=2):
    """Given an mdanalysis trajectory and contact distance cutoff, count the
    number of water molecules within cutoff of each residue"""
    protein = traj.select_atoms("protein")
    resids = np.unique(protein.resids)
    ret = []
    for rid in resids:
        wat = traj.select_atoms("resname WAT and around %f resid %d" % (cutoff, rid))
        ret.append((rid, len(wat)))
    return pd.DataFrame(ret, columns=("resid", "waters"))


def traj_frame_waters(i):
    """
    calls get_res_contacts and returns numpy area of contacts for that frame
    """
    curr_traj.trajectory[i]
    rc = get_res_waters(curr_traj)
    return rc.waters.to_numpy()


def make_waters_df(traj) -> DataFrame:
    """
    Creates and returns the final dataframe with water contacts averaged for
    each residue over the entire trajectory
    """
    waters = pd.DataFrame()
    protein = traj.select_atoms("protein")
    waters["resid"] = protein.residues.resids
    waters["resname"] = protein.residues.resnames
    print("Calculating waters...")
    pool = multiprocessing.Pool()
    traj_waters = pool.map(traj_frame_waters, range(curr_traj.trajectory.n_frames))
    traj_waters = np.array(traj_waters)
    waters["waters"] = traj_waters.mean(axis=0)
    return waters.set_index("resid")


if __name__ == "__main__":

    top = str(sys.argv[1])
    xtc = str(sys.argv[2])
    traj = read_traj(top, xtc)

    print("Calculating waters...")
    waters = make_waters_df(traj)

    print("Outputing csv...")
    waters.to_csv(sys.argv[3])

    print("Done.")
