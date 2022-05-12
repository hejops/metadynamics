#!/usr/bin/env python3
# from matplotlib import style
# import numpy as np
# import numpy as np
# import re
# import statistics
# import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd

maintitle = "FRET simulations (atoms 74, 420)"
xlabel = "Distance / Å"
ylabel = "dG / kJ mol-1"
# this is 2 args; needs to be 1
# labels = "Distance / Å", "dG / kJ mol-1"

temps = {
    "300": "blue",
    "350": "green",
    "400": "orange",
    "450": "yellow",
    "500": "red",
}


def group_results(df):

    # gives a good text summary
    # TODO: should energy also be included?
    # TODO: do t-test to determine if values -are- different
    # https://stackoverflow.com/a/26599490
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.first.html
    # mean, std, first etc are builtins, not strings!

    # min energy can be changed to opt as needed
    result = df.groupby(["temp"], as_index=False).agg(
        # {"max": ["mean", "std"], "opt": ["mean", "std"], "avg": ["mean", "std"]}
        {"min energy": ["mean", "std"]}
    )
    result.columns = [
        "temp",
        "min energy mean",
        "min energy std",
        # "opt dist mean",
        # "opt dist std",
        # "avg dist mean",
        # "avg dist std",
    ]
    result.reindex(columns=sorted(result.columns))

    return result


def make_bar(df, title):
    # produces a bar chart with error bars
    # TODO: subplots, sharex, etc

    # https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.plot.html
    # https://stackoverflow.com/a/38048077
    # https://stackoverflow.com/a/42131286
    # https://stackoverflow.com/a/57171666
    # https://towardsdatascience.com/the-quick-and-easy-way-to-plot-error-bars-in-python-using-pandas-a4d5cca2695d

    # for an actual histogram, see:
    # https://stackoverflow.com/a/11775745
    # https://stackoverflow.com/a/25539531
    # not useful for this data though!

    df.plot(
        kind="bar",
        rot=0,  # horizontal labels
        x="temp",
        ylabel="Distance / Å",
        ylim=(-100, 0),
        # y=["opt dist mean", "max dist mean", "avg dist mean"],
        # yerr=df[["opt dist std", "max dist std", "avg dist std"]].T.values,
        y=["min energy mean"],
        yerr=df[["min energy std"]].T.values,
        legend=True,
        title=title,
        color=["cornflowerblue", "seagreen", "purple"],
    )


if __name__ == "__main__":

    files = glob.glob("fes*dat")
    alltemps = {}

    # create a "master" dict to be turned into a df later
    for i, fes in enumerate(files):

        temp = f"{fes[3:6]} K"
        if "no_tmx" not in fes:
            temp += " + TMX"

        df = pd.read_csv(fes, delim_whitespace=True, comment="#", header=None)
        # temp, max dist, opt dist (min energy)
        # TODO: also add the average distance (of the entire run)
        # df[0].mean() ?
        alltemps[i] = [
            temp,
            df[0].max(),
            df[0][df[1].idxmin()],
            df[0].mean(),
            df[1].min(),
        ]

    print(alltemps)

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
    # https://stackoverflow.com/a/27975230

    newdf = pd.DataFrame.from_dict(
        alltemps, orient="index", columns=["temp", "max", "opt", "avg", "min energy"]
    )

    # selecting probably won't be necessary
    # tmx = newdf[~newdf["temp"].str.contains("no")]
    # tmx_group = group_results(tmx)
    # no_tmx = newdf[newdf["temp"].str.contains("no")]
    # no_tmx_group = group_results(no_tmx)
    # temp_300 = newdf[newdf["temp"].str.contains("300")]
    # group_300 = group_results(temp_300)
    # print(tmx_group)
    # print(no_tmx_group)
    # print(group_300)
    # make_bar(tmx_group, "TMX")
    # make_bar(no_tmx_group, "No TMX")
    # make_bar(group_300, "300 K")

    all_group = group_results(newdf)
    make_bar(all_group, "Temperature, TMX")
    print(all_group)

    # remove the "main" x label, but not the sublabels
    # https://stackoverflow.com/a/40707115
    # might break the plot
    # plt.axes().get_xaxis().get_label().set_visible(False)
    # plt.legend(["Optimal distance", "Maximum distance", "Average distance"])
    plt.legend(["Minimum energy"])
    plt.show()
