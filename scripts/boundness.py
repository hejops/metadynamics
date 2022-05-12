#!/usr/bin/env python3
# import logging
# import matplotlib.pyplot as plt
# import os
# import os
# import requests
# import shutil import urllib
from glob import glob
from natsort import natsorted
from os import link, getcwd
from os.path import dirname, abspath, relpath
from shutil import copyfile
import numpy as np
import pandas as pd
import re
import socket
import sys

import getpass
USER = getpass.getuser()

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# COLVAR = sys.argv[1]

# cutoff for residence time is set to 0.9
CUTOFF = 0.9


def read_colvar(COLVAR: str):
    with open(COLVAR) as f:
        first_line = f.readline().split()

    header = [x for x in first_line if x not in ["#!", "FIELDS"]]
    if "distK_Au" not in header:
        # print("No distK_Au column:", COLVAR)
        return

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table
    df = pd.read_csv(
        comment="#",
        delim_whitespace=True,
        filepath_or_buffer=COLVAR,
        header=0,
        names=header,
    )

    # round > 4 just returns standard form
    averages = df.mean(axis=0).round(3)

    # print(averages)

    # bound = df.mean(axis=0).round(3)

    r_time = len(df.loc[(df.distK_Au <= CUTOFF)]) / len(df)

    return averages


def read_xvg(XVG: str, residence: bool = False):
    """
    Run xtcdistances first!

    Works best with the new naming convention.
    Support for the old convention is hacky at best.
    """
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table

    # DIR = dirname(XVG)
    DIR = dirname(abspath(XVG))
    BIASES = {
        "nobias": "b",
        "coordf": "h",
        "funnel": "h",
        "dihG": "2",
        "0.3": "u",
        "_bound_": "b",
        "_unbound_": "u",
    }
    for b in BIASES:
        if b in DIR:
            boundness = BIASES[b]
            break
        # do not set a default!
        boundness = ""

    if not boundness:
        boundness = input("Bound? [b/u]")

    # TODO: this probably breaks with the new convention
    # if "bound_dist" in DIR:
    #     cvtype = "d"
    if DIR.endswith("dist"):
        cvtype = "d"
    elif DIR.endswith("stack"):
        cvtype = "s"
    elif DIR.endswith("dihedral"):
        cvtype = "h"
    elif DIR.endswith("coord"):
        cvtype = "c"

    df = pd.read_csv(
        comment="#",
        delim_whitespace=True,
        filepath_or_buffer=XVG,
        header=None,
        usecols=[1],
        # index_col=0,
        # squeeze = True
        # discard 1st column so that df becomes series
    )[1]

    assert df.shape == (2501,)

    name = XVG.replace(".xvg", "")
    # TODO: determine cv type and bias type from dirname
    print(name)

    if DIR.startswith("2021-12-"):

        # new set of runs contains run num, easy to work with
        # ckit1_tmx_npt_u1_atom...xvg
        run = name.split("_")[3][1]

    else:
        run = input("Run: ")

    res = name.split("_")[-1]
    if res == "715":
        res = "K"

    name = cvtype + boundness + run

    average = round(df.mean(), 3)
    # print(average)

    d = {
        # TODO use constants instead of strings
        "run": name,
        "residue": res,
        "avg dist": average,
    }

    if residence:
        residence_time = len(df.loc[(df <= CUTOFF)]) / len(df)
        # print(residence_time)
        d["r_time"] = round(residence_time, 3)

    return d


def batch_xvg():
    # xvgs = sys.argv[1:]
    # read in all xvg files in current directory
    xvgs = sorted(glob("*715.xvg"))

    print(f"{len(xvgs)} xvgs found")

    # in the new convention, every dir has 5 runs, each producing 5 xvgs (xtcdistances)
    # in the old convention, dir only has 1 or 2 runs, each with 5 xvgs

    # if len(xvgs) != 25:
    #     print("Not all xvgs have been generated; run xtcdistances first")
    #     sys.exit()

    # TODO: decide when to use residence; sys.arg?
    means = [read_xvg(f, residence=True) for f in xvgs]
    # means = [read_xvg(f) for f in xvgs]
    df = pd.DataFrame(means)

    # raw df format is:
    # cb1 K avg [r_time]
    # cb1 1 avg [r_time]
    # ...

    if "r_time" in df:
        # no need to pivot
        # keep rows with residue = K
        df = df[df.residue == "K"]
        df.drop("residue", axis=1, inplace=True)
        # print(df)
        piv = df

    else:
        # reshape to become: cb1 K 1 5 9 11

        # pivot always sorts
        # pivot_table sort=False is fake news
        # so just set column order manually
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
        col_order = ["K", "1", "5", "9", "11"]

        piv = df.pivot_table(
            index="run",
            columns="residue",
            values="avg dist",
            # values="r_time",
            # sort=True,
        ).reindex(col_order, axis=1)

    print(piv)
    df_to_table(piv)


def full_header(shortname: str) -> str:
    """
    Expects [cd][buh2][1-5], only reads 2nd character
    """
    if shortname == "diff":
        return "Difference"
    elif shortname[1] == "b":
        return "Bound"
    elif shortname[1] == "h":
        return "Half-bound"
    elif shortname[1] == "m":
        return "Multiple"
    elif shortname[1] == "u":
        return "Unbound"
    else:
        return shortname


def df_to_table(df: pd.DataFrame) -> None:
    if "diff" in df:
        df.drop("time", inplace=True)
        data_type = "cvs"

        # # stick to the cb1_cu1 convention for the lda dir,
        # # but use a cleaner name for dest dir
        # run = "_".join(list(df.columns)[:-1])
        # fname = f"{run}_{data_type}"
        # # # cb1_cu1 -> c_bu_1
        # # dest = re.sub(r"(\w)(\w)\d_\w(\w)(\d)", r"\1_\2\3_\4", run)
        # # this naming scheme is probably cleanest
        # # will probably retroactively apply to existing files with the old naming scheme
        # # cb1_cu1 -> c1_bu
        # # dest = re.sub(r"(\w)(\w)\d_\w(\w)(\d)", r"\1\4_\2\3", run)
        # # print(dest); sys.exit()

        # sim_type = cv type + bound state
        sim_type = "_".join(list(df.columns)[:-1])	# strip run number = cv type + boundness

        # replace abbreviated columns with full description
        # can be extended to the other conditions as more checks are added to full_header
        # https://stackoverflow.com/a/11354850
        # https://stackoverflow.com/a/16667215
        df.rename(columns=lambda x: full_header(x), inplace=True)
        print(df)

    elif "r_time" in df:
        # df.reset_index(drop=True, inplace=True)
        df = df.set_index("run")

        # TODO: determine based on...?
        data_type = "distances"
        # sim_type = "db"
        sim_type = list(df.index)[0][:-1]
        # print(sim_type);sys.exit()

    else:
        # pivoted df
        data_type = "distances"
        # "run" is not considered a column, but an index
        # run = df["run"][0][:-1]

        # run = list(df.index)[0][:-1]
        # fname = f"{run}_{data_type}"
        # dest = fname

        sim_type = list(df.index)[0][:-1]

    if socket.gethostname() == "oceanids":
        runs = "".join([x[2] for x in df.index])
        fname = f"{sim_type}{runs}_{data_type}"
    else:
        fname = f"{sim_type}_{data_type}"

    df.to_markdown(buf=f"{fname}.md", floatfmt=".3f")
    df.to_latex(buf=f"{fname}.tex")  # 3dp by default, apparently

    # if socket.gethostname() == "artemis":
    # hard link not allowed between /scratch and /home
    # d[buf2]

    for ext in ["md", "tex"]:
        if socket.gethostname() == "oceanids":
            data_type = "distances_unbiased"

        dest = f"/home/{USER}/gromacs/thesis/{data_type}/{fname}.{ext}"

        if os.path.isfile(dest):
            print(dest, "exists!")
            sys.exit()

        copyfile(f"{fname}.{ext}", dest)


if len(sys.argv) == 1:
    # TODO: detect cwd?
    # run from /scratch/user/npt/..., no args
    batch_xvg()

elif len(sys.argv) == 3:
    # run from /scratch/user/lda, 2 COLVAR
    # ene difference represents dG, sort of?
    avg1 = read_colvar(sys.argv[1])
    avg2 = read_colvar(sys.argv[2])
    col1 = sys.argv[1].replace(".COLVAR", "")
    col2 = sys.argv[2].replace(".COLVAR", "")
    merged = pd.concat(
        [avg1, avg2],
        axis=1,
        keys=[col1, col2],
    )
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    # add diff column
    merged["diff"] = abs(merged[col1] - merged[col2])
    df_to_table(merged)
