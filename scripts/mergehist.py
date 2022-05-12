#!/usr/bin/env python3

"""
Calculate free energy of a CV from multiple histograms (block-averaged).
Requires:

    hist.dat
    analysis.*.hist.dat

https://www.plumed.org/doc-v2.7/user-doc/html/masterclass-21-2-fes2.png
"""

# yes, needs to be matplotlib.pyplot; ignore pyright
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import sys


def weight_of(X, mu=0.6, sigma=0.5) -> float:
    """
    When X is raw data, energy range is nearly perfect (2.7 - 4.2, cf 2.8 -
    4.3), but the FES shape is completely distorted

    When X is weighted data, all weights are nearly equal and FES shape is
    somewhat incorrect. Energy range is also wrong

    X values further from mu get larger weight, which corresponds to the
    principle of metaD, i guess

    An awful lot of work for what amounts to basically nothing
    """
    foo = -((X - mu) ** 2) / (2 * (sigma ** 2))
    P = (math.e ** foo) / (math.sqrt(2 * math.pi) * sigma)
    weight = 1 / P
    return weight


weighted = int(sys.argv[1])
if weighted:
    print("Will calculate weights")

# TODO: use sys.argv[1] instead of hist
files = ["./hist.dat"] + sorted(glob.glob("./analysis.*.hist.dat"))

# N = number of histograms
# hist1[:, 1] = 2nd column of file
# histograms are, by definition, probability distributions
# each item in the array represents the probability of finding a value (X) in that bin (i = 0.00, 0.05, ...)
# for any given histogram, P(Xi) should add up to 1

N = sum_X = sum_X_sq = sum_w = sum_wX = 0

# programming equivalent of Sigma
for f in files:
    histo = np.loadtxt(f)
    bins = histo[:, 0]
    prob_dist = histo[:, 1]
    # print(histo.shape[1])

    N += 1
    sum_X += prob_dist
    sum_X_sq += prob_dist ** 2

    if weighted:

        # should probably be bins, not prob_dist
        weights = weight_of(bins)
        sum_w += weights
        sum_wX += prob_dist * weights

        # print(prob_dist)
        # print(values)
        # print(weights)
        # sys.exit()

# Final averages and variances
# these equations are more or less based on those found at the end of ex 3 (unweighted)
# \mu = \frac{1}{N} \sum X_i \qquad \qquad \sigma = \sqrt{ \frac{N}{N-1} \left[ \frac{1}{N} \sum_{i=1}^N X_i^2 - \left( \frac{1}{N}\sum_{i=1}^N X_i \right)^2 \right] }

assert N == len(files)

if weighted:
    print("sum wX", sum_wX)
    print("sum w", sum_w)
    mean = sum_wX / sum_w
else:
    print("sum X", sum_X)
    print(N)
    mean = sum_X / N

var = (N / (N - 1)) * ((sum_X_sq / N) - (mean ** 2))
error = np.sqrt(var / N)

print("mean", mean)

# sys.exit()

# ex 5, eq 1

# if weighted, 1st item in FES should be ~4.3

# note: np.log is actually ln!
fes = -np.log(mean)
print("FES", fes)

# Convert to error in fes
fes_err = error / (mean)

# And draw graph of free energy surface
plt.fill_between(histo[:, 0], fes - fes_err, fes + fes_err)
plt.xlabel("CV value")
plt.ylabel("Free energy")
plt.savefig("fes_avgblock.png")
# plt.show()
