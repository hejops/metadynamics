#!/usr/bin/env python3
# from matplotlib import style
# import glob
# import re
# import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

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


def group_results(file, name):

    # https://stackoverflow.com/a/26599490
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.GroupBy.first.html

    df = pd.read_csv(file, header=None)
    result = df.groupby(0, as_index=False).agg({1: ["mean", "std"]})
    # TODO: rename groups to full name?

    result.columns = [name, "Mean", "Std"]
    result = result.sort_values("Mean")

    make_bar(result, name + "s")
    # TODO merge columns in markdown (probably better with bash)
    print(result.to_markdown(index=False))

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
        x=0,
        ylabel="dG / kJ mol-1",
        ylim=(-30, 0),
        # y=["opt dist mean", "max dist mean", "avg dist mean"],
        # yerr=df[["opt dist std", "max dist std", "avg dist std"]].T.values,
        y=["Mean"],
        yerr=df[["Std"]].T.values,
        legend=False,
        title=title,
        color=["cornflowerblue"],
    )

    plt.axes().get_xaxis().get_label().set_visible(False)


def scatter(file):
    # meh
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv(file, header=None)
    # https://stackoverflow.com/a/44961301
    sns.regplot(x=df[0], y=df[1]).set(ylim=(-30, 0), xlim=(250, 500))
    # .xticks(np.arange(0, len(x)+1, 5))

    plt.xticks(ticks=[300, 350, 400, 450])


if __name__ == "__main__":

    # group_results("pose.csv", "Pose")
    # group_results("planar.csv", "Planar")
    # group_results("pistack.csv", "Pi-stack")

    scatter("temp.csv")

    plt.show()

    # remove the "main" x label, but not the sublabels
    # https://stackoverflow.com/a/40707115
    # might break the plot
    # plt.legend(["Optimal distance", "Maximum distance", "Average distance"])
    # plt.legend(["Minimum energy"])
    # plt.show()
