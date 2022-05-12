#!/usr/bin/env python3
# from glob import glob
# import logging
# import numpy as np
# import os
# import re
# import requests
# import shutil
# import urllib
from matplotlib import rc
import colorsys
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import sys

# usual setup, to be refactored
DPI = 100
IMG_WIDTH = 1920 * 2
font = {
    "family": "Times New Roman",
    "size": 60,
}
rc("font", **font)

fig = plt.figure()

weights_csv = sys.argv[1]

if "nostdev" in weights_csv:
    stdev_csv = weights_csv.replace("_nostdev.csv", "_cvs.csv")
else:
    # previously, weights with stdev applied were used, which is probably incorrect
    stdev_csv = weights_csv.replace(".csv", "_cvs.csv")

# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table
# contains only dx, weight
# weights are normalised to max (0-1), not sum
weights_df = pd.read_csv(
    comment="#",
    delim_whitespace=False,
    filepath_or_buffer=weights_csv,
    header=None,
    names=["feature", "weight"],
)

print(weights_df)

# contains dx, bound stdev, unbound stdev, diff (not used)
stdev_df = pd.read_csv(
    comment="#",
    delim_whitespace=False,
    filepath_or_buffer=stdev_csv,
    header=0,
    # names=header,
)

print(stdev_df)

merged_df = pd.concat(
    [stdev_df, weights_df.weight],
    axis=1,
).dropna()

merged_df.columns.values[0] = "feature"

merged_df["bound_stdev_weight"] = merged_df.Bound * merged_df.weight
merged_df["unbound_stdev_weight"] = merged_df.Unbound * merged_df.weight

# sum can be used in place of max (Parrinello uses sum)

merged_df["bound_stdev_weight"] = merged_df.bound_stdev_weight / max(
    merged_df.bound_stdev_weight
)
merged_df["unbound_stdev_weight"] = merged_df.unbound_stdev_weight / max(
    merged_df.unbound_stdev_weight
)

print(merged_df)

# TODO: feature + bound_stdev_weight -> bar plot

# 2 rows, 1 column
fig, ax = plt.subplots(2, 1)

# stacked plots are not very aesthetic
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html


def adjust_lightness(color: str, amount=0.8) -> tuple:
    # https://stackoverflow.com/a/49601444
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_colors(base_color: str) -> list:
    return [
        adjust_lightness(base_color)
        if int(x.replace("d", "")) < 5
        else f"tab:{base_color}"
        for x in merged_df.feature
    ]


ax[0].bar(
    merged_df.feature,
    merged_df.bound_stdev_weight,
    width=0.5,
    # color="tab:orange",
    color=plot_colors("orange"),
    # yerr=men_std,
    label="Bound",
)
ax[0].set(xlabel="Bound")

ax[1].bar(
    merged_df.feature,
    merged_df.unbound_stdev_weight,
    width=0.5,
    # color="tab:blue",  # match LDA output
    color=plot_colors("blue"),  # match LDA output
    # https://matplotlib.org/stable/tutorials/colors/colors.html
    # yerr=women_std,
    # bottom=men_means,
    label="Unbound",
)
ax[1].set(xlabel="Unbound")

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
# twiny/twinx not ideal, 2nd x label is placed at top
# secondary_xaxis not suitable, assumes numeric correspondence and requires reversible function

# ax2 = ax[0].twiny()  # instantiate a second axes that shares the same x-axis
# ax2.set_xlabel("Ligand / Virtual")  # we already handled the x-label with ax1

# secax = ax[0].secondary_xaxis(
#     "top",
#     functions=(deg2rad, rad2deg),
# )
# secax.set_xlabel("angle [rad]")

from matplotlib.offsetbox import AnchoredText

for i in [0, 1]:

    at = AnchoredText(
        "Ligand",
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[i].add_artist(at)

    at = AnchoredText(
        "Virtual",
        frameon=True,
        loc="upper right",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[i].add_artist(at)


# ax.set_ylabel("Weight")
fig.suptitle("Weighted standard deviation")
# fig.legend()

# plt.show()

# plt.bar(
#     merged_df.feature,
#     merged_df.bound_stdev_weight,
# )

fig.set_size_inches(IMG_WIDTH / float(DPI), (IMG_WIDTH * 9 / 16) / float(DPI))

plt.tight_layout()
plt.savefig(weights_csv.replace(".csv", ""))
