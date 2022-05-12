#!/usr/bin/env python3
# from glob import glob
# import logging
# import matplotlib.pyplot as plt
# import numpy as np
# import os
import pandas as pd

import getpass
USER = getpass.getuser()

# import re
# import requests
# import shutil
# import urllib
import sys

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# if __name__ == "__main__":
#     main()

f = "../thesis/fes_biased/FES_b1_bu"

f = sys.argv[1]

hills = f"/scratch/{USER}/deeplda_until_2022-01-11/3_post_lda/2021-12-22_ckit_bound_coord_sw=0.1_bu_every10,50ns/HILLS_b1"

# get min of free energy
# scan CV value in HILLS
# get frame

df = pd.read_csv(
    f,
    comment="#",
    header=None,
    delim_whitespace=True,
)

min_G = min(df[2])
CV = df.loc[df[2] == min_G][1].iloc[0]  # iloc returns the actual value

print(min_G, CV); sys.exit()

hills_df = pd.read_csv(
    # TODO: columns
    # time, sw, sigma, height, bias
    hills,
    comment="#",
    header=None,
    delim_whitespace=True,
    names=[
        "time",
        "CV",
        "sigma",
        "height",
        "biasfactor",
    ],
)

# print(hills_df)

# create new column: abs(x - CV)
# print(hills_df.loc[df[1] == CV])

hills_df["diff"] = abs(hills_df["CV"] - CV)

# print(CV)
# print(hills_df.diff)

CV_at_min = min(hills_df["diff"])

# print(CV_at_min)

frame = hills_df.loc[hills_df["diff"] == CV_at_min].time.iloc[0]

print(frame)
