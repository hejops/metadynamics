#!/usr/bin/env python3
# import csv
# import requests
# import urllib
from glob import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys

import getpass
USER = getpass.getuser()

# loadtxt doesn't work, but genfromtxt does with the exact same args???
# as long as a str is present, fields are stored as tuples, not "actual" 2D array
# np will remain a mystery to me
# https://stackoverflow.com/a/9534653


def csv_to_dict(f: str):
    data = np.genfromtxt(
        f,
        delimiter=",",
        usecols=(1, 2),
        dtype=None,  # let strings be strings; note: decimal not supported
        encoding=None,
    )

    return {x[0]: x[1] for x in data}


all_dict = {}

for f in glob(f"/scratch/{USER}/colvar/d*.csv"):
    for CV, rank in csv_to_dict(f).items():
        # print(CV, rank)

        if CV in all_dict:
            all_dict[CV] += rank
        else:
            all_dict[CV] = rank

all_dict = dict(sorted(all_dict.items(), key=lambda x: x[1], reverse=True))

for cv, rank in all_dict.items():
    print(cv, rank)
