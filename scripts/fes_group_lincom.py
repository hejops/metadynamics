#!/usr/bin/env python3
# from glob import glob
# from pprint import pprint
# import logging
# import numpy as np
# import os
# import re
# import requests
# import shutil
# import urllib
import sys

from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

DPI = 100
IMG_WIDTH = 1920 * 2
# anything smaller than 40 is not recommended
font = {
    "family": "Times New Roman",
    "size": 40,
}
rc("font", **font)
# rc('text', usetex = True)


def main(files: list) -> None:

    # files = sorted(glob(f"{FESC_DIR}/hb2*.FES"))
    # files = sorted(glob(f"{FESC_DIR}/sb5*.FES"))

    data = []
    for f in files:

        df = pd.read_csv(
            f,
            delim_whitespace=True,
            comment="#",
            usecols=[0, 1],
            header=None,
            names=["CV", "G"],
        )

        # print(df);sys.exit()
        # db1_du1_distAu_tet01.FES
        # name = re.sub(r".+/.{3}_.{3}_([^.]+)\.FES", r"\1", f)

        # remove PBC
        # df = df.loc[df["CV"] < 7]

        data.append(
            {
                # "CV": name,
                # "CV type": determine_CV_type(name),
                "x": df["CV"],
                "G": df["G"],
            }
        )

    merged_x = pd.concat(
        # [x for x in data],
        [x["x"] for x in data],
        # ignore_index=True,
        # sort=False,
    )

    Y_MIN = min(pd.concat([x["G"] for x in data]))
    Y_MAX = max(pd.concat([x["G"] for x in data]))
    X_MIN = min(pd.concat([x["x"] for x in data]))
    X_MAX = max(pd.concat([x["x"] for x in data]))
    # Y_MAX = max(merged[1])

    # refer to ./fretgraph.py for how to set up subplots
    fig, subplot = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # fig.suptitle(MAIN_TITLE)

    # list required for .index
    # TODO: set comp makes order random smh
    # sorted is an ok compromise
    # CV_TYPES = list(sorted({x["CV type"] for x in data}))
    # print(CV_TYPES)

    for item in data:

        # CV_type = item["CV type"]
        # CV_type = "distance"
        # df = item["array"]
        x = item["x"]
        G = item["G"]

        # idx = CV_TYPES.index(CV_type)
        # row = COORD_DICT[idx][0]
        # col = COORD_DICT[idx][1]

        row = col = 0
        # print(CV_type, row, col)

        # row, col = 0, 0

        # print(df[1])
        # print(df[1] - max(df[1]))
        # # sys.exit()

        subplot[row, col].plot(
            # plt.plot(
            x,
            G,
            # G - Y_MAX,  # set 0 at top
            # color=TITLE_DICT[CV_type][1],
            # label=CV_type,
        )

        # above
        # subplot[row, col].set_title(TITLE_DICT[CV_type][0])

        # below
        # xlabel = MAIN_LABEL

        # using italic reverts to ugly font, too bad
        subplot[row, col].set(xlabel=r"Combination CV, $\it{s}$")
        subplot[row, col].set(ylabel="ΔG / kJ mol¯¹")

    # TODO: allow setting via sys.argv

    plt.xlim([X_MIN, X_MAX])
    plt.ylim([Y_MIN, Y_MAX])

    # plt.xlim([-50, 50])
    plt.xlim([-2, 0])
    # plt.xlim([-10, 30])
    # plt.xlim([-20, 0])

    # # plt.ylim([-120, 0])

    plt.tight_layout()

    DPI = fig.get_dpi()
    fig.set_size_inches(IMG_WIDTH / float(DPI), (IMG_WIDTH * 9 / 16) / float(DPI))

    # # TODO: work on filename
    # outname = "_".join([x.split("/")[-1].replace(".FES", "") for x in files])

    # FES_bN_hu
    outname = files[0].split("_")[-1]
    print("Writing to", outname)

    # savefig must always be called before show
    plt.savefig(
        f"./{outname}.png",
        bbox_inches="tight",
    )


files = sys.argv[1:]
print(f"{len(files)} files selected")
main(files)
