#!/usr/bin/env python3
# import logging
# import requests
# import urllib
from glob import glob
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import sys

import getpass
USER = getpass.getuser()

# from fretgraph import make_subplot
from fes_2cv import G_LABEL
from mkboxplot import determine_CV_type

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

DPI = 100
IMG_WIDTH = 1920 * 2
# anything smaller than 40 is not recommended
font = {
    "family": "Times New Roman",
    "size": 40,
}
rc("font", **font)

# too lazy to turn int to grid-coord
# for i in (0,1,2,3):
#     print(bin(i))
#     print("{0:b}".format(i))
# sys.exit()

PBC_CUTOFF = 7

# only for 2D (> 2x2)
COORD_DICT = {
    0: (0, 0),
    1: (1, 0),
    2: (1, 0),
    3: (1, 1),
}

# TODO: eventually convert to subdict
TITLE_DICT = {
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "<++>": ["<++>", "<++>"],
    "dihG02": ["Tetrad G2 dihedrals", "green"],
    "dihT1": ["TMX dihedrals", "green"],
    "dist1": ["TMX-tetrad (COM)", "orange"],
    "distA1": ["A1-tetrad distances", "brown"],
    "distA2": ["A1-TMX distances", "green"],
    "distA3": ["A1-TMX distances", "green"],
    "distA4": ["A1-TMX distances", "green"],
    "distA5": ["A1-TMX distances", "green"],
    "distA6": ["A1-TMX distances", "green"],
    "distAu": ["Au-tetrad distances", "orange"],
    "distC1": ["TMX-tetrad distances", "blue"],
    "distR": ["TMX-exterior distances", "blue"],
    "distG02": ["Tetrad G2 distances (COM)", "green"],
    "distH": ["Tetrad hydrogen bond lengths", "green"],
    "distK": ["K distances", "red"],
    "distX": ["Random distances (DNA-DNA)", "black"],
    "stackG02": ["Tetrad G2 stacking", "purple"],
    "stackT1": ["TMX stacking", "purple"],
}

SIM_DICT = {
    "hb": {
        "desc": "Dihedral CVs by type",
        "label": "Dihedral / rad",
    },
    "db3": {
        "desc": "Distance CVs by type",
        "label": "Distance / nm",
    },
    "sb": {
        "desc": "Distance (COM) CVs by type",
        "label": "Distance / nm",
    },
}

SIM_TYPE = "db3"
MAIN_TITLE = SIM_DICT[SIM_TYPE]["desc"]
MAIN_LABEL = SIM_DICT[SIM_TYPE]["label"]

FES_DIR = f"/home/{USER}/gromacs/thesis/fes_multi_unbiased"
FESC_DIR = f"/home/{USER}/gromacs/thesis/fes_unbiased"
STATS_DIR = f"/home/{USER}/gromacs/pres/stats"


def read_fes(selected_cvs: list) -> tuple[list, str]:
    """
    Do not deal with min/max here as it leads to unnecessary complexity
    """

    files = sorted(glob(f"{FES_DIR}/{SIM_TYPE}*.FES"))
    # print(len(files)); sys.exit()

    fname = SIM_TYPE + "_" + "_".join(selected_cvs)

    # files = [f for f in files if f[0].split("/")[-1].split("_")[2].startswith(selected)]

    files = [f for f in files if any(x in f for x in selected_cvs)]

    # print(files)
    # sys.exit()

    data = []
    for f in files:
        df = pd.read_csv(
            f,
            delim_whitespace=True,
            comment="#",
            header=None,
            names=["CV", "G"],
        )
        # db1_du1_distAu_tet01.FES
        name = re.sub(r".+/.{3}_.{3}_([^.]+)\.FES", r"\1", f)

        # remove PBC
        df = df.loc[df["CV"] < PBC_CUTOFF]

        # print(df)
        # print(f, df.loc[df["CV"] < 7]); sys.exit()

        data.append(
            {
                "CV": name,
                "CV type": determine_CV_type(name),
                "x": df["CV"],
                "G": df["G"],
            }
        )

    return data, fname


def plot_fes(selected_cvs: list) -> None:
    """
    Stacked graphs
    Separate subplot for each CV type
    """

    data, fname = read_fes(selected_cvs)

    # print(data)
    # sys.exit()

    merged_x = pd.concat(
        # [x for x in data],
        [x["x"] for x in data],
        # ignore_index=True,
        # sort=False,
    )

    Y_MAX = max(pd.concat([x["G"] for x in data]))
    # Y_MAX = max(merged[1])

    # refer to ./fretgraph.py for how to set up subplots
    fig, subplot = plt.subplots(
        nrows=len(selected_cvs),
        ncols=1,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # fig.suptitle(MAIN_TITLE)

    # list required for .index
    # TODO: set comp makes order random smh
    # sorted is an ok compromise
    CV_TYPES = list(sorted({x["CV type"] for x in data}))
    # print(CV_TYPES)

    for item in data:

        CV_type = item["CV type"]
        # df = item["array"]
        x = item["x"]
        G = item["G"]

        idx = CV_TYPES.index(CV_type)

        # row = COORD_DICT[idx][0]
        # col = COORD_DICT[idx][1]

        row = idx
        col = 0

        print(CV_type, row, col)

        # row, col = 0, 0

        # print(df[1])
        # print(df[1] - max(df[1]))
        # # sys.exit()

        subplot[row, col].plot(
            # plt.plot(
            x,
            G - Y_MAX,  # set 0 at top
            color=TITLE_DICT[CV_type][1],
            label=CV_type,
        )

        # CV description above plot
        subplot[row, col].set_title(TITLE_DICT[CV_type][0], pad=20)

        # label bottom-most plot only
        if row == len(selected_cvs) - 1:
            xlabel = MAIN_LABEL
            subplot[row, col].set(xlabel=xlabel)

        subplot[row, col].set(ylabel=G_LABEL)

    plt.xlim([min(merged_x), max(merged_x)])
    plt.ylim([-Y_MAX, 0])

    # this may lead to squished plots
    # plt.tight_layout()

    DPI = fig.get_dpi()
    fig.set_size_inches(IMG_WIDTH / float(DPI), (IMG_WIDTH * 9 / 16) / float(DPI))

    # savefig must always be called before show
    plt.savefig(f"{FES_DIR}/{fname}")

    # plt.show()


for sel in [
    ["distAu", "distC1", "distA1"],
    ["distH"],
    # ["distH", "distAu"],
    # ["distC1", "distA1"],
    # ["distA1", "distAu"],
    # ["distC1", "distH"],
    # ["distR", "distX"],
]:
    plot_fes(sel)


def read_stats(f: str) -> pd.DataFrame:
    # db1_du1_distAu_tet01.FES
    name = re.sub(r".+/.{3}_.{3}_([^.]+)\.STATS", r"\1", f)

    # print(name)
    # print(CV_type)
    # use cols, with header; then easy to process

    df = pd.read_csv(
        f,
        delim_whitespace=True,
        # comment="#",
        header=None,
        usecols=[1, 2],  # time, mean, var
        names=["mean", "var"],
    )
    df["CV"] = name
    SD = np.sqrt(df["var"])
    df["SD"] = SD
    df["SEM"] = SD / np.sqrt(50000)

    # df["COL"] returns a series
    # whatever
    if int(df["SEM"]) > int(df["mean"]):
        df["SEM"] = 0

    return df


def plot_stats():
    """
    Horizontal bar graph with error bars (only if SD < mean)
    All CVs in one plot

    https://blogs.sas.com/content/iml/2019/10/09/statistic-error-bars-mean.html
    https://www.graphpad.com/guides/prism/latest/statistics/statwhentoplotsdvssem.htm
    https://www.graphpad.com/support/faq/is-it-better-to-plot-graphs-with-sd-or-sem-error-bars-answer-neither/
    https://www.investopedia.com/ask/answers/042415/what-difference-between-standard-error-means-and-standard-deviation.asp

    SEM equals SD / sqrt(N)
    CLM is a multiple of the SEM, usually 1.96

    Turns out SEM almost always ends up quite small (sometimes basically 0),
    because of large N (50000?)
    """

    files = sorted(glob(f"{STATS_DIR}/{SIM_TYPE}*.STATS"))
    # files = [f for f in files if "distH" in f or "distAu" in f]

    data = pd.concat([read_stats(f) for f in files])

    # if var > mean, different color, remove error bar
    my_color = np.where(
        # data["var"] > data["mean"],
        data["SD"] == 0,
        "orange",  # condition false
        "skyblue",  # condition true
    )

    # https://www.python-graph-gallery.com/185-lollipop-plot-with-conditional-color/
    # https://pythonforundergradengineers.com/python-matplotlib-error-bars.html
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html

    fig, ax = plt.subplots(
        figsize=(IMG_WIDTH / float(DPI), (IMG_WIDTH * 9 / 16) / float(DPI)),
        dpi=DPI,
    )
    ax.barh(
        data["CV"],
        data["mean"],
        xerr=data["SEM"],
        color=my_color,
        # label="1",
        # title="Distance CVs",
        # align="center",
        # alpha=0.5,
        # ecolor="black",
        # capsize=10,
    )

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title(MAIN_TITLE)
    ax.set_xlabel(MAIN_LABEL)

    plt.savefig(f"{STATS_DIR}/{SIM_TYPE}")
    # plt.show()


# plot_stats()
