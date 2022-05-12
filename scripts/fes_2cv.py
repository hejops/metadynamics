#!/usr/bin/env python3
# from glob import glob
# from mpl_toolkits import mplot3d
# import logging
# import matplotlib.pyplot as plt
# import os
# import re
# import requests
# import shutil
# import urllib
from matplotlib import cm
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d
from pprint import pprint
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess as sp
import sys

# https://stackoverflow.com/a/50332785
rcParams["axes.labelpad"] = 50

# TODO: get 2D projection along each CV

# usual setup, to be refactored
DPI = 100
IMG_WIDTH = 1920 * 2
font = {
    "family": "Times New Roman",
    "size": 80,
    # font size is exaggerated because 2D plots are usually subfig (half-width)
}
rc("font", **font)
fig = plt.figure()
fig.set_size_inches(IMG_WIDTH / float(DPI), (IMG_WIDTH * 9 / 16) / float(DPI))
G_LABEL = r"ΔG / kJ mol $^-$¹"  # a nonsensical mix of unicode and tex

DIST_LABEL = "K-Au distance (z) / nm"

# X, Y, Z = axes3d.get_test_data(0.05)
# print(X.shape, Y.shape, Z.shape); sys.exit()
# # shape must be (x, x), where x > 1


def read_hills(hills: str) -> pd.DataFrame:

    cols = [0, 1]
    names = ["time", "sw"]
    with open(hills) as f:
        firstline = f.readline().rstrip()
        NUM_CVs = firstline.count("sigma")
        if NUM_CVs > 1:
            cols.append(2)
            names.append("zdist")

    return pd.read_csv(
        hills,
        comment="#",
        header=None,
        delim_whitespace=True,
        usecols=cols,
        names=names,
    )


def read_fes(fes: str) -> pd.DataFrame:

    cols = [0, 1]
    names = ["sw", "G"]
    with open(fes) as f:
        firstline = f.readline().rstrip()
        NUM_CVs = firstline.count("der")
        if NUM_CVs > 1:
            cols.append(2)
            names = ["sw", "zdist", "G"]

    #! FIELDS [ sw cyl.z file.free ] der_sw der_cyl.z
    return pd.read_csv(
        fes,
        delim_whitespace=True,
        comment="#",
        header=None,
        usecols=cols,
        names=names,
    )


def plot_and_crop_image(img: str) -> None:
    # idk how to get rid of white space, so just trim with imagemagick convert
    # plt.tight_layout()
    # ax.axis('tight')
    # only needed for 3d plot

    plt.savefig(img)
    sp.call(
        f"convert {img}.png -trim {img}.png",
        shell=True,
    )
    print("Wrote", img)


def get_local_minima(
    df: pd.DataFrame,
    CV: str,
) -> list:
    """get_local_minima.

    Does not detect inflexion points

    Args:
        df (pd.DataFrame): df
        CV (str): name of column in df

    Returns:
        list:
    """
    # argrelextrema works better on array than df
    array = np.array(df.G)
    minima_G = array[argrelextrema(array, np.less)]
    # print(minima_G)
    CV_at_minima = df[df.G.isin(minima_G)]  # return the subset of original df
    return CV_at_minima[CV]  # select only the CV column


def get_frame(
    hills_df: pd.DataFrame,
    sw: float,
    zdist: float = 0,
) -> None:
    """get_frame.

    Search for frame closest to given value(s) of sw (and zdist)

    Args:
        hills_df (pd.DataFrame): hills_df
        sw (float): sw
        zdist (float): zdist

    Returns:
        None:
    """
    # print("Searching for frame with:")

    # create the diff columns
    # these columns are unique to sw and zdist at function call!

    if zdist:
        hills_df["sw_diff"] = abs(hills_df["sw"] - sw)
        hills_df["zdist_diff"] = abs(hills_df["zdist"] - zdist)
        hills_df["diff"] = hills_df["sw_diff"] + hills_df["zdist_diff"]

    else:
        zdist = 0
        hills_df["diff"] = abs(hills_df["sw"] - sw)

    # print(hills_df)

    min_diff = min(hills_df["diff"])
    best_frame = hills_df.loc[hills_df["diff"] == min_diff]

    # divide by 10 to get vmd frame
    vmd_frame = int(best_frame.time.iloc[0] / 10)
    print(
        vmd_frame,
        round(sw, 3),
        round(zdist, 3),
        sep=",",
    )


def main(fes: str) -> tuple:

    if not fes.startswith("FES"):
        print("Not a FES:", fes)
        sys.exit()

    # if os.path.isfile(f + ".png"):
    #     print("Already done:", f)
    #     return

    df = read_fes(fes)
    X = df.sw
    Y = df.zdist
    Z = df.G

    # print(Z)

    min_G = min(Z)
    print("# Energy minimum:", min_G)

    # print(min_G); sys.exit()

    sw_at_min: float = df.loc[df["G"] == min_G].sw.iloc[0]
    zdist_at_min: float = df.loc[df["G"] == min_G].zdist.iloc[0]

    # get a 2D slice of the 3D data that contains the global energy minimum
    df_sw_slice = df.loc[df["sw"] == sw_at_min]
    df_zdist_slice = df.loc[df["zdist"] == zdist_at_min]

    # TODO: this should probably go somewhere else
    # get all energy minima
    # for sw, select only first and last
    minima_sw = get_local_minima(df_zdist_slice, "sw")
    minima_sw = minima_sw[:: len(minima_sw) - 1]
    minima_zdist = get_local_minima(df_sw_slice, "zdist")

    # TODO: construct a dict of energy minima, to be passed to plot_2d

    # print(minima_sw)
    # print(minima_zdist)
    # sys.exit()

    find_best_hills_frames(
        fes,
        sw_at_min,
        minima_sw,
        zdist_at_min,
        minima_zdist,
    )

    # due to the ordering of rows, data is not treated as a "function"
    # x1 y, x1 y, ..., x1 y, x2 y, x2 y, ...
    # this produces discontinuous lines instead of continuous plots
    # this can be fixed by sorting by sw first
    sorted_df = df.sort_values(
        [
            "sw",
            "zdist",
        ],
        ascending=(True, True),
    )

    plot_2d(
        df=df,
        sliced_df=df_zdist_slice,
        xlabel="s",
        cv="sw",
        minima=minima_sw,
    )

    plot_2d(
        df=sorted_df,
        sliced_df=df_sw_slice,
        xlabel="K-Au distance (z) / nm",
        cv="zdist",
        minima=minima_zdist,
    )

    return min_G, sw_at_min


def plot_2d(
    df: pd.DataFrame,
    sliced_df: pd.DataFrame,
    xlabel: str,
    cv: str,
    minima,
):
    plt.clf()
    ax = fig.add_subplot(111)

    # identify the bin containing the global energy minimum,
    # and show/highlight that bin in the plot
    # x1-y, x2-y, ..., xn-y, repeat (with new zdist)
    plt.plot(
        df[cv],
        df.G,
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        sliced_df[cv],
        sliced_df.G,
        color="green",
        linewidth=5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(G_LABEL)

    # TODO: range-based approach (tuple) is better
    poses = {
        -0.3: "B1",
        -0.7: "B2",
        -1.2: "U1a",
        0.3: "U2",
        1.0: "B1",
        1.4: "U1b",
        1.7: "U1c",
    }

    # https://matplotlib.org/stable/tutorials/text/annotations.html
    for x in minima:
        y = sliced_df.loc[df[cv] == x].G.iloc[0]
        print(x, y)
        approx_x = round(x, 1)
        if approx_x not in poses:
            continue
        ax.annotate(
            # TODO: annotation placement is not ideal, but good enough
            poses[approx_x],
            xy=(x, y),
            xycoords="data",
            textcoords="data",
            # xytext=(0.8, 0.95),
            # arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="right",
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.savefig(fes + "_" + cv)


def find_best_hills_frames(
    fes: str,
    sw_at_min: float,
    minima_sw: list,
    zdist_at_min: float = 0,
    minima_zdist: list = [],
):
    """
    Iterate through a list of local minima to obtain frame information on each
    minimum

    Calls get_frame()

    """

    # attempt to find best frames in HILLS
    hills = fes.replace("FES", "HILLS")
    if os.path.isfile(hills):

        hills_df = read_hills(hills)

        print("# Global minimum: (frame, sw, zdist)")
        get_frame(hills_df, sw_at_min, zdist_at_min)

        print("# Local minima:")

        # reduce 2 loops to 1
        foo = [(sw, zdist_at_min) for sw in minima_sw]
        bar = [(sw_at_min, zdist) for zdist in minima_zdist]

        # print(foo)
        # print(bar)

        for sw, zdist in foo + bar:
            if sw == sw_at_min and zdist == zdist_at_min:
                continue
            get_frame(hills_df, sw, zdist)

        # TODO: sp call, open vmd at given frame
        # vmd -e <(echo "animate goto $FRAME") "$PDB" # &

        # sys.exit()


def plot_3d():

    # print(sw_at_min, zdist_at_min)

    # Z2 = np.array([[x] for x in df.G])  # force "2D" array out of 1D data; required for some 3D functions

    # print(X.shape, Y.shape, Z.shape); sys.exit()

    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    # labels are shortened, no unit
    ax.set_xlabel("s")
    ax.set_ylabel("d")
    ax.set_zlabel("ΔG")
    ax.tick_params(axis="z", which="major", pad=20)

    # ax.grid(False)
    # plt.grid(b=None)

    # 3d deprecated in favour of gnuplot

    # # ax.plot_wireframe(
    # plt.tricontourf(
    #     X,
    #     Y,
    #     Z,
    #     # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #     # most color maps look better when reversed
    #     cmap=plt.cm.get_cmap("plasma" + "_r"),
    # )
    # # plt.plot_surface(X, Y, Z)
    # plot_and_crop_image(fes)
    # sys.exit()

    # plt.show()

    # ax.autoscale(False)

    # TODO: disable grid
    # TODO: better color map (red -> blue)

    # plt.plot(
    #     # xys_bad[:, 0],
    #     # xys_bad[:, 1],
    #     # color="r",
    #     linestyle="None",
    #     # markersize=10.0,
    # )

    # cset = ax.contour(X, Y, Z2, cmap=cm.coolwarm)
    # ax.clabel(cset, fontsize=9, inline=1)

    # TODO: refactor these 2 plot sections


if __name__ == "__main__":
    Gs = []
    sw = []
    for fes in sys.argv[1:]:
        print(fes)
        min_G, sw_at_min = main(fes)
        Gs.append(min_G)
        sw.append(sw_at_min)
        print()

    print(np.mean(Gs), np.mean(sw))
