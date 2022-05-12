#!/usr/bin/env python3
# import requests
# import urllib
from glob import glob
from matplotlib import rc

import getpass
USER = getpass.getuser()

# from scipy.interpolate import spline
from natsort import natsorted
from scipy.interpolate import make_interp_spline, BSpline
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import sys

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# import matplotlib.font_manager as font_manager
# font_manager._rebuild()

# if newly installed font not found (ubuntu cringe), delete ~/.cache/matplotlib
# font = {"family": "CMU Sans Serif", "size": 22}
# font size will depend on image size and dpi
# for 3840x2160, 100 dpi, use font size 40
font = {"family": "Times New Roman", "size": 40}
rc("font", **font)

"""
Show values of multiple CVs in 1 simulation as a boxplot (median, IQR).

https://en.wikipedia.org/wiki/Box_plot

>1 simulation not supported (yet?)

Also consider violin plots

Note that boxplots do not make any statistical assumptions about the underlying
distribution (e.g. mean). As such, skewed distributions will produce many
"outliers".

    Bottom line, a boxplot is not a suitable outlier detection test but rather
    an exploratory data analysis to understand the data. While boxplots do
    identify extreme values, these extreme values are not truely outliers, they
    are just values that outside a distribution-less metric on the near
    extremes of the IQR.

    https://swampthingecology.org/blog/too-much-outside-the-box-outliers-and-boxplots/
"""

CV_DICT = {}
PBC_CUTOFF = 7  # .5


def calc_dG(fes: str) -> float:
    data = np.loadtxt(fes)
    print(data)
    if data.any():
        Gs = data[:, 1]
        Gs = Gs[np.isfinite(Gs)]  # nanmax doesn't actually remove inf, sad
        dG = round(max(Gs) - min(Gs), 1)
    else:
        dG = 0
    return dG


def values_to_array_dG(file: str) -> tuple:
    """
    Read VALUES file, return as array

    No need to worry about how the VALUES file is generated (wrt column
    selection, specifically @3); that's handled by PLUMED (cv_to_fes)
    """
    name = file.split("/")[-1].replace(".VALUES", "")

    # array = np.random.rand(100)
    array = np.loadtxt(fname=file, usecols=1)

    # extremely large distances are probably caused by PBC jumps and should be dropped
    # i haven't verified in vmd since big trrs take so long to load

    # 7 is probably a better cutoff, but whatever
    if max(array) > PBC_CUTOFF:
        print("Warning: PBC detected")

    fes = file.replace(".VALUES", ".FES")
    dG = calc_dG(fes)

    return name, np.extract(array < PBC_CUTOFF, array), dG


def determine_CV_type(name: str) -> str:
    # split by space, then underscore, then remove digits (cringe)
    # TODO: change to re.sub
    CV_type = name.split(" ")[0].split("_")[0]

    # group together, don't split
    # H = tetrad H bond, R = random inter, X = random intra
    CV_type = re.sub(fr"(coord|[dHRX])\d+", fr"\1", CV_type)

    return CV_type


def boxplots():

    BASE_DIR = f"./{sys.argv[1]}"

    # if len(glob(f"{BASE_DIR}/*.png")) >= len(CV_DICT):
    #     print("Already processed:", BASE_DIR)
    #     sys.exit()

    # TODO: remove pngs in BASE_DIR?

    CVs = {}

    if sys.argv[1].startswith("h"):
        xlabel = "Dihedral / rad"
    else:
        xlabel = "Distance / nm"

    for f in sorted(glob(f"{BASE_DIR}/*.VALUES")):

        print(f)
        name, array, dG = values_to_array_dG(f)

        CV_type = determine_CV_type(name)

        if CV_type not in CV_DICT:
            CV_DICT[CV_type] = {}

        # print(name); sys.exit()
        CV_DICT[CV_type][f"{name} ({dG})"] = {
            "array": array,
            "dG": dG,
        }

        print("Processed", f)
        print(name, "->", CV_type)
        print()

        print(CV_DICT)
    # sys.exit()

    # TODO: renormalise values (make all medians equal)
    # https://gawron.sdsu.edu/python_for_ss/course_core/book_draft/visualization/boxplot.html

    for CV_type, CVs in CV_DICT.items():

        if not CVs:
            print("No CVs in", CV_type)
            continue

        # sort nested dict by sub-value
        # CV with largest dG shown on top, smallest dG at bottom
        # https://www.geeksforgeeks.org/python-sort-nested-dictionary-by-key/

        # just sort alpha for now
        CVs = dict(natsorted(CVs.items(), reverse=True))

        # # SORT = False
        # if CV_type in ["d", "coord"]:
        #     # coord = alphabetical
        #     CVs = dict(natsorted(CVs.items(), reverse=True))
        # else:
        #     CVs = dict(sorted(CVs.items(), key=lambda x: x[1]["dG"]))

        # https://stackoverflow.com/a/52274064
        # https://matplotlib.org/stable/gallery/pyplots/boxplot_demo_pyplot.html
        fig, ax = plt.subplots()

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html#matplotlib.axes.Axes.boxplot
        # boxplot takes a list of arrays
        # TODO: fix x-axis, probably 0-5?
        ax.boxplot(
            x=[x["array"] for x in CVs.values()],
            labels=CVs.keys(),
            notch=True,
            vert=False,
        )
        # ax.set_yticklabels(CVs.keys())

        # plt.xticks(rotation=45, ha="right")

        # max num of CVs = 96
        # for this to be (barely) visible in 1 image, use size 32:24, dpi 1200

        ax.set(xlabel=xlabel)

        DPI = fig.get_dpi()
        fig.set_size_inches(3840.0 / float(DPI), 2160.0 / float(DPI))

        png = f"{BASE_DIR}/{CV_type}_{len(CVs)}.png"
        print("Writing", png)

        # TODO: distX not written? file exists, but cannot be opened
        plt.savefig(
            png,
            # absurdly large dpi prevents image from getting squashed, hopefully
            # 2400 takes very long to write/open
            # dpi=800,
            dpi=100,
            bbox_inches="tight",
        )

        # clear figure after write? doesn't seem to actually work
        # https://stackoverflow.com/a/21884375
        plt.clf()

        # sys.exit()

        # plt.show()


def smooth_data(x, y):
    # https://stackoverflow.com/a/5284038
    # 300 represents number of points to make between T.min and T.max
    # makes curves unnaturally round
    xnew = np.linspace(x.min(), x.max(), 250)
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    ynew = spl(xnew)
    return xnew, ynew


def normal_plots():
    """
    FES -> graph

    Strictly limited to 1 CV type
    """

    BASE_DIR = f"/scratch/{USER}/cv_analysis/k4"

    fig, ax = plt.subplots()
    for f in glob(f"{BASE_DIR}/distAu*.FES"):
        FES = np.loadtxt(f)
        # x, y = smooth_data(FES[:, 0], FES[:, 1])
        x, y = FES[:, 0], FES[:, 1]
        plt.plot(
            x,
            y,
            # label=f,
        )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    boxplots()

# normal_plots()
