#!/usr/bin/env python3
# from matplotlib import style
# import numpy as np
# import re
from matplotlib import rc
import glob
import matplotlib.font_manager
import matplotlib.pyplot as plt
import pandas as pd
import sys

maintitle = "Free energies of pseudo-FRET simulations (DT3:N1 - DT14:N1)"
xlabel = "Distance / Å"
ylabel = "dG / kJ mol-1"
# this is 2 args; needs to be 1
# labels = "Distance / Å", "dG / kJ mol-1"

temps = {
    "300": "blue",
    "350": "green",
    "400": "orange",
    "450": "red",
    "500": "violet",
}

font = {"family": "CMU Sans Serif", "size": 22}
rc("font", **font)


def read_1cv_fes(fes: str) -> pd.DataFrame:
    #! FIELDS [ sw cyl.z file.free ] der_sw der_cyl.z
    return pd.read_csv(
        fes,
        delim_whitespace=True,
        comment="#",
        header=None,
        usecols=[0, 1],
        names=["sw", "G"],
    )


# df = read_1cv_fes("../thesis/fes_final/FES_b1_every10,50ns,1cv_sigma=005")
# df = read_1cv_fes("../thesis/fes_final/FES_b1_every10,50ns,1cv")
for fes in sys.argv[1:]:
    df = read_1cv_fes(fes)
    min_G = min(df.G)
    sw_at_min: float = df.loc[df["G"] == min_G].sw.iloc[0]
    print(fes, min_G, sw_at_min)

sys.exit()

# plt.close("all")

# turning point https://gist.github.com/ben741/d8c70b608d96d9f7ed231086b237ba6b
# https://www.delftstack.com/howto/matplotlib/how-to-add-title-to-subplots-in-matplotlib/
# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html


def make_subplot(fes: str, subplot):

    df = pd.read_csv(
        fes,
        delim_whitespace=True,
        comment="#",
        header=None,
    )

    # shouldn't need regex if files are consistently named
    # temp = re.search("[0-9]{3}", fes).group(0)
    temp = fes[3:6]
    idx = foundtemps.index(temp)

    # place no TMX on the left
    if "no_tmx" in fes:
        title = f"no TMX"
        tmx = 0
    else:
        title = f"TMX"
        tmx = 1

    # df.plot(0, 1)	    # pandas plots are drawn immediately and cannot be combined!

    # swap the dimensions and ignore tmx state
    if wide:
        tmx = idx
        idx = 0

    subplot[idx, tmx].plot(df[0], df[1], color=temps[temp])
    if "300" in fes:
        subplot[idx, tmx].set_title(title)
    if "500" in fes:
        subplot[idx, tmx].set(xlabel=xlabel)
    subplot[idx, tmx].set(ylabel=f"{temp} K\nΔG / kJ mol-1")


if __name__ == "__main__":

    files = glob.glob("fes*dat")

    # 1x5
    if len(sys.argv) == 2 and "tmx" in sys.argv[1]:
        wide = True
        files = [x for x in files if "no_tmx" not in x]
        foundtemps = sorted(list(set([x[3:6] for x in files])))
        fig, subplot = plt.subplots(
            1,
            len(foundtemps),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

    # 1 row, 2 columns
    elif len(sys.argv) == 2:
        wide = False
        files = [x for x in files if sys.argv[1] in x]
        # this doesn't need to exist
        foundtemps = [x[3:6] for x in files]
        fig, subplot = plt.subplots(
            1,
            2,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

    # 5x2
    else:
        wide = False
        foundtemps = sorted(list(set([x[3:6] for x in files])))
        fig, subplot = plt.subplots(
            len(foundtemps),
            2,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

    # fig.suptitle(maintitle)
    for fes in files:
        make_subplot(fes, subplot)

    plt.xlim([0, 50])
    plt.ylim([-90, 10])
    # plt.legend()

    # plt.show()

    filename = f"fret_{'_'.join(foundtemps)}.png"

    # >closing as wontfix
    # https://github.com/matplotlib/matplotlib/issues/2305/#issuecomment-22766670
    DPI = fig.get_dpi()
    fig.set_size_inches(3840.0 / float(DPI), 2160.0 / float(DPI))

    plt.savefig(filename)
