#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[TUTORIAL] Data-driven collective variables for enhanced sampling.ipynb

https://colab.research.google.com/drive/1dG0ohT75R-UZAFMf_cbYPNQwBaOsVaAA

*Data-driven collective variables for enhanced sampling* Luigi Bonati*, Valerio
Rizzi and Michele Parrinello, J. Phys. Chem. Lett., 11, 2998-3004 (2020)
[doi:10.1021/acs.jpclett.0c00535](http://doi.org/10.1021/acs.jpclett.0c00535).

![alt text](https://drive.google.com/uc?id=1fxFCJWY6UWXxyNheIv4N9PPx3ouB7kID)

NOTE: This code has been tested with Pytorch version **1.4/1.5** and Libtorch
C++ API version **1.4**.

Pytorch version 1.9.0 has been known to cause errors with PLUMED!

Two COLVAR files are expected (bound/unbound), with frames in the rows and CVs
(input / output) in the columns.

Large sections of this file have been modified for local use.

Wall clock time (CVs, frames):
    < 3 min (285 distance, 100k)
"""

# @title Load modules

# !pip3 install torch==1.5.0 torchvision==0.6.0

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import progressbar
import scipy
import sys
import time
import torch

from timeit import default_timer as timer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from matplotlib import rc

# font = {"family": "Times New Roman", "size": 12}
# rc("font", **font)

# TODO: increase horiz spacing between subplots
plt.rcParams["font.family"] = "Times New Roman"

# TODO: the model performance is probably summarised in lda.dat -- figure out how to interpret it

# from google.colab import files

torch.manual_seed(0)
# random.seed(0)
np.random.seed(0)

start = timer()

print("Using Pytorch", torch.__version__)

if sys.stdin.isatty():
    print("No display detected; plots will not be shown.")

# Define NN architecture and loss function {{{

##################################
# Custom Dataset
##################################
class ColvarDataset(Dataset):
    """COLVAR dataset"""

    def __init__(self, colvar_list):
        self.nstates = len(colvar_list)
        self.colvar = colvar_list

    def __len__(self):
        return len(self.colvar[0])

    def __getitem__(self, idx):
        x = ()
        for i in range(self.nstates):
            x += (self.colvar[i][idx],)
        return x


# useful for cycling over the test dataset if it is smaller than the training set
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


##################################
# Define Networks
##################################


class NN_DeepLDA(nn.Module):
    def __init__(self, l):
        super(NN_DeepLDA, self).__init__()

        modules = []
        for i in range(len(l) - 1):
            print(l[i], " --> ", l[i + 1], end=" ")
            if i < len(l) - 2:
                modules.append(nn.Linear(l[i], l[i + 1]))
                modules.append(nn.ReLU(True))
                print("(relu)")
            else:
                modules.append(nn.Linear(l[i], l[i + 1]))
                print("")

        self.nn = nn.Sequential(*modules)

        # norm option
        self.normIn = False

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.Mean = Mean
        self.Range = Range

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)

        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

        return x.sub(Mean).div(Range)

    def get_hidden(
        self, x: Variable, svd=False, svd_vectors=False, svd_eigen=False, training=False
    ) -> (Variable):
        if self.normIn:
            x = self.normalize(x)
        z = self.nn(x)
        return z

    def set_lda(self, x: torch.Tensor):
        self.lda = nn.Parameter(x.unsqueeze(0), requires_grad=False)

    def get_lda(self) -> (torch.Tensor):
        return self.lda

    def apply_lda(self, x: Variable) -> (Variable):
        z = torch.nn.functional.linear(x, self.lda)
        return z

    def forward(self, x: Variable) -> (Variable):
        z = self.get_hidden(x, svd=False)
        z = self.apply_lda(z)
        return z

    def get_cv(self, x: Variable) -> (Variable):
        return self.forward(x)


# auxiliary class to export a model which outputs the topmost hidden layer
class NN_Hidden(nn.Module):
    def __init__(self, l):
        super(NN_Hidden, self).__init__()

        modules = []
        for i in range(len(l) - 1):
            if i < len(l) - 2:
                modules.append(nn.Linear(l[i], l[i + 1]))
                modules.append(nn.ReLU(True))
            else:
                modules.append(nn.Linear(l[i], l[i + 1]))

        self.nn = nn.Sequential(*modules)

        # norm option
        self.normIn = False

    def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
        self.normIn = True
        self.Mean = Mean
        self.Range = Range

    def normalize(self, x: Variable):
        batch_size = x.size(0)
        x_size = x.size(1)
        Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
        Range = self.Range.unsqueeze(0).expand(batch_size, x_size)
        return x.sub(Mean).div(Range)

    def get_hidden(
        self, x: Variable, svd=False, svd_vectors=False, svd_eigen=False, training=False
    ) -> (Variable):
        if self.normIn:
            x = self.normalize(x)
        z = self.nn(x)
        return z

    def forward(self, x: Variable) -> (Variable):
        z = self.get_hidden(x, svd=False)
        return z


##################################
# Loss function
##################################


def LDAloss_cholesky(H, label, test_routines=False):
    # sizes
    N, d = H.shape

    # Mean centered observations for entire population
    H_bar = H - torch.mean(H, 0, True)
    # Total scatter matrix (cov matrix over all observations)
    S_t = H_bar.t().matmul(H_bar) / (N - 1)
    # Define within scatter matrix and compute it
    S_w = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    S_w_inv = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    buf = torch.Tensor().new_zeros((d, d), device=device, dtype=dtype)
    # Loop over classes to compute means and covs
    for i in range(categ):
        # check which elements belong to class i
        H_i = H[torch.nonzero(label == i).view(-1)]
        # compute mean centered obs of class i
        H_i_bar = H_i - torch.mean(H_i, 0, True)
        # count number of elements
        N_i = H_i.shape[0]
        if N_i == 0:
            continue
        S_w += H_i_bar.t().matmul(H_i_bar) / ((N_i - 1) * categ)

    S_b = S_t - S_w

    S_w = S_w + lambdA * torch.diag(
        torch.Tensor().new_ones((d), device=device, dtype=dtype)
    )

    ## Generalized eigenvalue problem: S_b * v_i = lambda_i * Sw * v_i

    # (1) use cholesky decomposition for S_w
    # NOTE: deprecated in newer versions;
    # torch.cholesky -> torch.linalg.cholesky
    L = torch.cholesky(S_w)  # , upper=False)

    # (2) define new matrix using cholesky decomposition and
    L_t = torch.t(L)
    L_ti = torch.inverse(L_t)
    L_i = torch.inverse(L)
    S_new = torch.matmul(torch.matmul(L_i, S_b), L_ti)

    # (3) solve  S_new * w_i = lambda_i * w_i

    eig_values, eig_vectors = torch.symeig(S_new, eigenvectors=True)

    # L, V = torch.symeig(A, eigenvectors=True)
    # should be replaced with
    # L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')

    # # upper was previously the default; now it must be specified
    # eig_values, eig_vectors = torch.linalg.eigh(S_new, UPLO="U")

    eig_vectors = eig_vectors.t()
    # (4) sort eigenvalues and retrieve old eigenvector
    # eig_values, ind = torch.sort(eig_values, 0, descending=True)
    max_eig_vector = eig_vectors[-1]
    max_eig_vector = torch.matmul(L_ti, max_eig_vector)
    norm = max_eig_vector.pow(2).sum().sqrt()
    max_eig_vector.div_(norm)

    loss = -eig_values[-1]

    return loss, eig_values, max_eig_vector, S_b, S_w


# Evaluate LDA over all the training set
def check_LDA_cholesky(loader, model):
    # gradient calculation is disabled
    # https://pytorch.org/docs/stable/generated/torch.no_grad.html

    with torch.no_grad():
        for data in loader:
            X, y = data[0].float().to(device), data[1].long().to(device)
            H = model.get_hidden(X)
            _, eig_values, eig_vector, _, _ = LDAloss_cholesky(H, y)

    return eig_values, eig_vector


# }}}

# Encoding and plotting functions {{{

##################################
# Encoding functions
##################################


def encode_hidden(loader, model, batch, n_hidden, device):
    """Compute the hidden layer over a dataloader"""
    s = np.empty((len(loader), batch, n_hidden))
    l = np.empty((len(loader), batch))
    for i, data in enumerate(loader):
        x, lab = data[0].float(), data[1].long()
        x = Variable(x).to(device)
        cv = model.get_hidden(x, svd=False)
        # cv = model.apply_pca(cv)
        s[i] = cv.detach().cpu().numpy()
        l[i] = lab

    s = s.reshape(len(loader) * batch, n_hidden)
    s = s[0 : len(loader) * batch]

    l = l.reshape(len(loader) * batch)
    l = l[0 : len(loader) * batch]

    sA = s[l == 0]
    sB = s[l == 1]

    return sA, sB


def encode_cv(loader, model, batch, n_cv, device):
    """Compute the CV over a dataloader"""
    s = np.empty((len(loader), batch, n_cv))
    l = np.empty((len(loader), batch))
    for i, data in enumerate(loader):
        x, lab = data[0].float(), data[1].long()
        x = Variable(x).to(device)
        cv = model(x)
        s[i] = cv.detach().cpu().numpy()
        l[i] = lab

    s = s.reshape(len(loader) * batch, n_cv)
    s = s[0 : len(loader) * batch]

    l = l.reshape(len(loader) * batch)
    l = l[0 : len(loader) * batch]

    sA = s[l == 0]
    sB = s[l == 1]

    return sA, sB


def encode_cv_all(loader, model, batch, n_cv, device):
    """Compute the CV over a dataloader with labels"""
    s = np.empty((len(loader), batch, n_cv))
    l = np.empty((len(loader), batch))
    for i, data in enumerate(loader):
        x, lab = data[0].float(), data[1].long()
        x = Variable(x).to(device)
        cv = model.get_cv(x)
        s[i] = cv.detach().cpu().numpy()
        l[i] = lab

    s = s.reshape(len(loader) * batch, n_cv)
    s = s[0 : len(loader) * batch]

    l = l.reshape(len(loader) * batch)
    l = l[0 : len(loader) * batch]

    return s, l


##################################
# Plotting functions
##################################


def plot_results(save=False, testing=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_training(axes[0], save)
    plot_H(axes[1], save, testing)
    plot_CV(axes[2], save, testing)
    if save:
        fig.savefig(f"{outname}_training.png", dpi=150)
        plt.close()
    # else:
    #     plt.show()


def plot_training(ax, save=False, training=False):
    ax.set_title("Deep-LDA optimization")
    # ax.plot(np.asarray(ep), np.asarray(eig), ".-", c="tab:green", label="train-batch")
    # ax.plot(np.asarray(ep), np.asarray(eig_t), ".-", c="tab:grey", label="train")
    # ax.plot(np.asarray(ep), np.asarray(eig_val), ".-", c="tab:orange", label="valid")
    # move back eig, etc to cpu for plotting
    ax.plot(
        np.asarray(epochs),
        np.asarray([x.cpu().detach().numpy() for x in eig]),
        ".-",
        c="tab:green",
        label="train-batch",
    )
    ax.plot(
        np.asarray(epochs),
        np.asarray([x.cpu().detach().numpy() for x in eig_t]),
        ".-",
        c="tab:grey",
        label="train",
    )
    ax.plot(
        np.asarray(epochs),
        np.asarray([x.cpu().detach().numpy() for x in eig_val]),
        ".-",
        c="tab:orange",
        label="valid",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1st Eigenvalue")
    ax.legend()


def plot_H(ax, save=False, testing=False):
    ax.set_title("LDA on Hidden-space H")
    # -- Testing and Validation histograms --
    trA, trB = encode_hidden(train_all_loader, model, train_data, n_hidden, device)
    eigen = max_eig_vector.cpu().detach().numpy()

    ax.scatter(trA[:, 0], trA[:, 1], c="tab:red", label="train A", alpha=0.3)
    ax.scatter(trB[:, 0], trB[:, 1], c="tab:blue", label="train B", alpha=0.3)

    if testing:
        ttA, ttB = encode_hidden(valid_loader, model, valid_data, n_hidden, device)
        ax.scatter(
            ttA[:, 0], ttA[:, 1], c="tab:orange", label="valid A", s=0.2, alpha=0.5
        )
        ax.scatter(
            ttB[:, 0], ttB[:, 1], c="tab:cyan", label="valid B", s=0.2, alpha=0.5
        )
        mIN = np.min(
            [np.min(trA[:, 0]), np.min(trB[:, 0]), np.min(ttA[:, 0]), np.min(ttB[:, 0])]
        )
        mAX = np.max(
            [np.max(trA[:, 0]), np.max(trB[:, 0]), np.max(ttA[:, 0]), np.max(ttB[:, 0])]
        )
    else:
        mIN = np.min([np.min(trA[:, 0]), np.min(trB[:, 0])])
        mAX = np.max([np.max(trA[:, 0]), np.max(trB[:, 0])])

    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")

    # x=np.linspace(mIN,mAX,100)
    # y=-eigen[0]/eigen[1]*x+0
    # plt.plot(x,y, linewidth=2, label='DeepLDA')
    ax.legend()


def plot_CV(ax, save=False, testing=False):
    sA, sB = encode_cv(train_all_loader, model, train_data, n_cv, device)
    sA, sB = sA[:, 0], sB[:, 0]
    if testing:
        stA, stB = encode_cv(valid_loader, model, valid_data, n_cv, device)
        stA, stB = stA[:, 0], stB[:, 0]
        min_s = np.min([np.min(sA), np.min(sB), np.min(stA), np.min(stB)])
        max_s = np.max([np.max(sA), np.max(sB), np.max(stA), np.max(stB)])
    else:
        min_s = np.min([np.min(sA), np.min(sB)])
        max_s = np.max([np.max(sA), np.max(sB)])

    b = np.linspace(min_s, max_s, 100)

    ax.set_title("Deep-LDA CV Histogram")
    ax.hist(
        sA,
        bins=b,
        ls="dashed",
        alpha=0.7,
        lw=2,
        color="tab:red",
        label="train A",
        density=True,
    )
    ax.hist(
        sB,
        bins=b,
        ls="dashed",
        alpha=0.7,
        lw=2,
        color="tab:blue",
        label="train B",
        density=True,
    )

    if testing:
        plt.hist(
            stA,
            bins=b,
            ls="dashed",
            alpha=0.5,
            lw=2,
            color="tab:orange",
            label="valid A",
            density=True,
        )
        plt.hist(
            stB,
            bins=b,
            ls="dashed",
            alpha=0.5,
            lw=2,
            color="tab:cyan",
            label="valid B",
            density=True,
        )

    ax.legend()


# }}}

# **Load files** {{{

# Specify dataset structure (use **from_column=1** to exclude time from COLVAR file)

first_col = 1  # 0-indexed; ignore time column

# fileA = "./OAH_G1/COLVAR_B_example"
# fileB = "./OAH_G1/COLVAR_U_example"

# fileA = "../../scratch/colvar_ckit/db1"
# fileB = "../../scratch/colvar_ckit/du1"

fileA = sys.argv[1]
fileB = sys.argv[2]

outname = f"{fileA}_{fileB}"

if "b" not in fileA:
    print("First file must be bound state")
    sys.exit()

with open(fileA) as f:
    first_line = f.readline().split()

# energy is actually not required
# @3 is still unknown

if "@3" in first_line:
    # this is a mistake
    # this includes @3, omits last CV
    # i maintain it here for "legacy reasons"
    # will be removed once i figure out the best way to sort it all out
    header = [x for x in first_line if x not in ["#!", "FIELDS", "time", "ene"]]
    header.pop()
else:
    # this omits @3, includes last CV
    header = [x for x in first_line if x not in ["#!", "FIELDS", "time", "@3", "ene"]]

# header = [x for x in first_line if x not in ["#!", "FIELDS", "time", "@3", "ene"]]

header_string = ",".join(header)
num_inputs = len(header)
# print(num_inputs); sys.exit()

print(f"Loading files: {fileA}, {fileB}")

col_range = range(first_col, first_col + num_inputs)

print(len(col_range), num_inputs)

assert len(col_range) == num_inputs

distA = np.loadtxt(fileA, usecols=col_range)
distB = np.loadtxt(fileB, usecols=col_range)

print("[Imported data]")
print("- shape:", distA.shape)

assert distA.shape == distB.shape

"""## **Create datasets**

- **standardize_inputs**: normalize each inputs such that it assumes values from -1 to 1. This is achieved by computing the mean and the range of the values over the training set.

- Choose the training set size (**train_data**) and the batch size (**batch_tr**). The batch size should be large enough to give estimates of the covariance matrices which are representative of the population for the computation of LDA.

- A validation dataset (**valid_data**)) is also defined, but it is evaluated in a single batch.
"""

##@title Create datasets
standardize_inputs = True  # @param {type:"boolean"}

if standardize_inputs:
    print("[Standardize inputs]")
    print("- Calculating mean and range over the training set")
    Max = np.amax(np.concatenate([distA, distB], axis=0), axis=0)
    Min = np.amin(np.concatenate([distA, distB], axis=0), axis=0)

    Mean = (Max + Min) / 2.0
    Range = (Max - Min) / 2.0
    if np.sum(np.argwhere(Range < 1e-6)) > 0:
        print(
            "- [Warning] Skipping normalization where range of values is < 1e-6. Input(s):",
            np.argwhere(Range < 1e-6).reshape(-1),
        )
        Range[Range < 1e-6] = 1.0
else:
    Mean = 0
    Range = 0

# create labels
lA = np.zeros_like(distA[:, 0])
lB = np.ones_like(distB[:, 0])

dist = np.concatenate([distA, distB], axis=0)
dist_label = np.concatenate([lA, lB], axis=0)

p = np.random.permutation(len(dist))
dist, dist_label = dist[p], dist_label[p]

FRAMES = distA.shape[0]

# @title Training and validation set size
# train_data = 16000  # @param {type:"integer"}

# train_data not a very descriptive name tbh
train_data = int(FRAMES * 0.8)

batch_tr = int(FRAMES * 0.2)  # @param {type:"integer"}
train_labels = ColvarDataset([dist[:train_data], dist_label[:train_data]])
train_loader = DataLoader(train_labels, batch_size=batch_tr, shuffle=True)

# create additional dataset which cover all the training set in one batch
train_all_labels = ColvarDataset([dist[:train_data], dist_label[:train_data]])
train_all_loader = DataLoader(train_all_labels, batch_size=train_data)

# The validation is evaluated in a single batch
valid_data = int(FRAMES * 0.2)  # @param {type:"integer"}
batch_val = valid_data
valid_labels = ColvarDataset(
    [
        dist[train_data : train_data + valid_data],
        dist_label[train_data : train_data + valid_data],
    ]
)
valid_loader = DataLoader(valid_labels, batch_size=batch_val)

# don't proceed if any of the loaders end up empty
# the numbers are generally 4 1 1

assert len(train_loader) >= 1
assert len(train_all_loader) >= 1
assert len(valid_loader) >= 1

# }}}

# **NN and training parameters** {{{

"""

Inizialize the neural network and the optimizer with the following parameters:
- **hidden_nodes**: NN architecture, specify the number of nodes per hidden layer. The last layer correspond to the space where LDA is performed. 
- **lrate**: learning rate of the optimizer (in this case, ADAM)
- **lambdA**: $S_w$ regularization
- **l2_reg**: L2 weight regularization

"""

##@title NN and training parameters
# type
dtype = torch.float32
# wheter to use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

categ = 2
n_cv = 1

# hidden_nodes = "30,30,5"  # @param {type:"raw"}
hidden_nodes = "10,8,6,4"  # @param {type:"raw"}
nodes = [int(x) for x in hidden_nodes.split(",")]
nodes.insert(0, num_inputs)
n_hidden = nodes[-1]

print("[NN Architecture]")
print("- hidden layers:", nodes)
print("")
print("========= NN =========")
model = NN_DeepLDA(nodes)
if standardize_inputs:
    model.set_norm(
        torch.tensor(Mean, dtype=dtype, device=device),
        torch.tensor(Range, dtype=dtype, device=device),
    )
print("======================")
model.to(device)

# if torch.cuda.is_available():
#     print("using CUDA acceleration")
#     print("========================")

# -- Optimization --
lrate = 0.001  # @param {type:"slider", min:0.0001, max:0.005, step:0.0001}
lambdA = 0.05  # @param {type:"number"}
l2_reg = 1e-5  # @param {type:"number"}
act_reg = 2.0 / lambdA  # lorentzian regularization

print("")
print("[Optimization]")
print("- Learning rate \t=", lrate)
print("- l2 regularization \t=", l2_reg)
print("- lambda (S_w reg.) \t=", lambdA)
print("- lorentian (CV reg.) \t=", act_reg)

# OPTIMIZERS
opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=l2_reg)

# define arrays and values
epochs = []
eig = []
eig_t = []
eig_val = []
init_epoch = 0
best_result = 0
# }}}

# **Training** {{{
"""

Set the number of epochs (**num_epochs**) and how often print the optimization details (**print_loss**) and plot the results (**plot_every**).

Notes: 
- during the training only the hidden representation h_1 vs h_2 is shown.
- if the cell is executed multiple times the training continues. To reset the NN run again the **NN and training parameters** cell.
"""

##@title Training
num_epochs = 50  # @param {type:"number"}
print_loss = 1  # @param {type:"slider", min:1, max:100, step:1}
plot_every = 25  # @param {type:"slider", min:1, max:100, step:1}
plot_validation = True  # @param {type:"boolean"}

# format output
float_formatter = lambda x: "%.6f" % x
np.set_printoptions(formatter={"float_kind": float_formatter})

print(
    "[{:>3}/{:>3}] {:>10} {:>10} {:>10} {:>10}".format(
        "ep", "tot", "eig_train", "eig_valid", "reg loss", "eigenvector"
    )
)

# -- Training --
for epoch in range(num_epochs):
    for data in train_loader:
        # =================get data===================
        X, y = data[0].float().to(device), data[1].long().to(device)
        # =================forward====================
        # H,S = model.get_hidden(X,svd=True,svd_eigen=True)
        H = model.get_hidden(X)
        # =================lda loss===================
        lossg, eig_values, max_eig_vector, Sb, Sw = LDAloss_cholesky(H, y)
        model.set_lda(max_eig_vector)
        s = model.apply_lda(H)
        # =================reg loss===================
        reg_loss = H.pow(2).sum().div(H.size(0))
        reg_loss_lor = -act_reg / (1 + (reg_loss - 1).pow(2))
        # =================backprop===================
        opt.zero_grad()
        lossg.backward(retain_graph=True)
        reg_loss_lor.backward()
        opt.step()

    # Compute LDA over entire datasets and save LDA eigenvector
    train_eig_values, train_eig_vector = check_LDA_cholesky(train_all_loader, model)
    model.set_lda(train_eig_vector)
    valid_eig_values, valid_eig_vector = check_LDA_cholesky(valid_loader, model)

    # save results
    epochs.append(epoch + init_epoch + 1)
    eig.append(eig_values[-1])
    eig_t.append(train_eig_values[-1])
    eig_val.append(valid_eig_values[-1])

    if (epoch + 1) % print_loss == 0:
        print(
            "[{:3d}/{:3d}] {:10.3f} {:10.3f} {:10.3f} ".format(
                init_epoch + epoch + 1,
                init_epoch + num_epochs,
                # copy GPU tensor to CPU, detach it from the graph, then convert to numpy array
                # train_eig_values.detach().numpy()[-1],
                train_eig_values.cpu().detach().numpy()[-1],
                valid_eig_values.cpu().numpy()[-1],
                reg_loss,
            ),
            train_eig_vector.cpu().numpy(),
        )

    # only show intermittent plots when graphical environment present
    if not sys.stdin.isatty():
        if (epoch + 1) % plot_every == 0:
            plot_results(testing=plot_validation)

init_epoch += num_epochs
# }}}

# **Inspect hidden and CV space** {{{
"""
Here we investigate some properties about the hidden layer and the CV.
1.   Plot the hidden components $h$ for the training set 
2.   Plot a scatter plot between each pair of them. Here we report also the projection 
3.   Export all the hidden components / CV calculated on the training set for further external analysis
4.   Print mean and std.dev. for the CV in the two basins, useful for setting the parameters of enhanced sampling

Obviously, the plots are not shown if no graphical environment is present.
"""

# @title Plot hidden components
# encode_hidden computes the hidden variables for an entire dataset, using the specified model
trA, trB = encode_hidden(train_all_loader, model, train_data, n_hidden, device)

fig, axs = plt.subplots(1, n_hidden, figsize=(3.5 * n_hidden, 3.5))

fig.suptitle("Hidden components")
for i in range(n_hidden):
    ax = axs[i]
    ax.set_xlabel("Training set")
    ax.set_ylabel(r"$h_" + str(i) + "$")
    ax.plot(trA[:, i], c="tab:red", label="trA")
    ax.plot(trB[:, i], c="tab:blue", label="trB")
    ax.legend()

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# @title Scatter plot w/LDA boundaries
from itertools import combinations

# get LDA eigenvector
eigen = model.get_lda().cpu().numpy()[0]

# compute h
trA, trB = encode_hidden(train_all_loader, model, train_data, n_hidden, device)
ttA, ttB = encode_hidden(valid_loader, model, valid_data, n_hidden, device)

# plot layout
n_plots = len(list(combinations(range(n_hidden), 2)))
num_plots_per_line = 5  # @param {type:"slider", min:1, max:10, step:1}
num_lines = int(n_plots / num_plots_per_line + 0.999)

# define subplots
fig, axs = plt.subplots(
    num_lines,
    num_plots_per_line,
    figsize=(3 * num_plots_per_line, 3 * num_lines),
    dpi=100,
)
axs = axs.reshape(-1)
fig.suptitle("Scatter plots of hidden components")

# iterate and plot
idx = 0
point_size = 2  # @param {type:"slider", min:0.5, max:20, step:0.5}
for i, j in combinations(range(n_hidden), 2):
    ax = axs[idx]
    idx += 1
    ax.scatter(trA[:, i], trA[:, j], c="tab:red", s=point_size, label="trA")
    ax.scatter(
        ttA[:, i],
        ttA[:, j],
        c="tab:orange",
        s=point_size,
        label="testA",
        alpha=0.3,
        marker="o",
    )
    ax.scatter(trB[:, i], trB[:, j], c="tab:blue", s=point_size, label="trB")
    ax.scatter(
        ttB[:, i],
        ttB[:, j],
        c="tab:cyan",
        s=point_size,
        label="testB",
        alpha=0.3,
        marker="o",
    )
    # plot LDA line
    mIN = np.min(
        [np.min(trA[:, i]), np.min(trB[:, i]), np.min(ttA[:, i]), np.min(ttB[:, i])]
    )
    mAX = np.max(
        [np.max(trA[:, i]), np.max(trB[:, i]), np.max(ttA[:, i]), np.max(ttB[:, i])]
    )
    x = np.linspace(mIN, mAX, 100)
    y = -eigen[i] / eigen[j] * x + 0
    ax.plot(
        x,
        y,
        linewidth=2,
        label="LDA boundary",
        color="darkgrey",
        alpha=0.7,
        linestyle="dashed",
    )
    # labels
    ax.set_xlabel(r"$h_" + str(i) + "$")
    ax.set_ylabel(r"$h_" + str(j) + "$")
    if idx == 1:
        leg = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=5,
            borderaxespad=0,
            frameon=False,
        )
        leg.set_in_layout(False)

fig.tight_layout(rect=[0, 0.03, 1, 0.93])
# plt.show()

# @title Export CV and hidden variables
# TODO: idk what these files are for
export_hidden = True  # @param {type:"boolean"}
export_cv = True  # @param {type:"boolean"}

if export_hidden:
    trA, trB = encode_hidden(train_all_loader, model, train_data, n_hidden, device)
    np.savetxt(
        f"{outname}_hidden_traj_A.dat",
        np.transpose([trA[:, i] for i in range(trA.shape[1])]),
    )
    np.savetxt(
        f"{outname}_hidden_traj_B.dat",
        np.transpose([trB[:, i] for i in range(trB.shape[1])]),
    )

if export_cv:
    trA, trB = encode_cv(train_all_loader, model, train_data, n_cv, device)
    np.savetxt(
        f"{outname}_cv_traj_A.dat",
        np.transpose([trA[:, i] for i in range(trA.shape[1])]),
    )
    np.savetxt(
        f"{outname}_cv_traj_B.dat",
        np.transpose([trB[:, i] for i in range(trB.shape[1])]),
    )

sA, sB = encode_cv(train_all_loader, model, train_data, n_cv, device)
sA, sB = sA[:, 0], sB[:, 0]


print("=========== STATE A ===========")
print("- Mean   :", np.mean(sA))
print("- DevStd :", np.std(sA))
print("=========== STATE B ===========")
print("- Mean   :", np.mean(sB))
print("- DevStd :", np.std(sB))
print("===============================")

"""## **Features ranking**

Compute feature importances $r_i$ by summing the modulus of the weights $w_{ij}$ of the connections between each input $i$ and each node $j$ of the first layer: 

$r_i =\sigma_i \sum_j ^{n_1}| w_{ij} ^{(1)} |$

The importances are multiplied by the standard deviations of the inputs $\sigma_i$ (disable **multiply_by_stddev** to see just the sum of the weights). In addition they are normalized such that $\sum_i r_i=1$. 

Plot setup: 
- The inputs can be ordered by the feature importances (**order_by_importance**) 
- Each feature is labelled with a number from 1 to n_inputs. In alternative, the **input names** can be specified (by enabling **use_input_names**).

**[TODO]** add doc for derivatives setup
"""

ranking_type = "weights"  # @param ["weights", "derivatives"]

multiply_by_stddev = True  # @param {type:"boolean"}

order_by_importance = True  # @param {type:"boolean"}
use_input_names = True  # @param {type:"boolean"}

# input_names = "d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12"  # @param {type:"string"}
# input_names = "d_1-2,d_1-3,d_1-4,d_1-5,d_1-6,d_1-7,d_1-8,d_1-9,d_1-10,d_1-11,d_2-3,d_2-4,d_2-5,d_2-6,d_2-7,d_2-8,d_2-9,d_2-10,d_2-11,d_3-4,d_3-5,d_3-6,d_3-7,d_3-8,d_3-9,d_3-10,d_3-11,d_4-5,d_4-6,d_4-7,d_4-8,d_4-9,d_4-10,d_4-11,d_5-6,d_5-7,d_5-8,d_5-9,d_5-10,d_5-11,d_6-7,d_6-8,d_6-9,d_6-10,d_6-11,d_7-8,d_7-9,d_7-10,d_7-11,d_8-9,d_8-10,d_8-11,d_9-10,d_9-11,d_10-11" #@param {type:"string"}

if use_input_names:
    # in_name = input_names.split(",")
    input_names = np.asarray(header)
else:
    input_names = np.arange(1, num_inputs + 1)

in_num = np.arange(num_inputs)
input_ranks = np.zeros(num_inputs)

# compute std to correct for different magnitudes
if multiply_by_stddev:
    if standardize_inputs:
        dist2 = (dist - Mean) / Range
    else:
        dist2 = dist
    in_std = np.std(dist2, axis=0)

# compute by summing the weights
if ranking_type == "weights":
    for i in range(num_inputs):
        input_ranks[i] = model.nn[0].weight[:, i].abs().sum().item()

# compute by derivatives w.r.t. inputs
# this seems to freeze? so we just use weights
elif ranking_type == "derivatives":
    for data in train_all_loader:
        X = Variable(data[0].float().to(device), requires_grad=True)
        scv = model.get_cv(X)
        for j in range(0, train_data, 10):
            scv[j].backward(retain_graph=True)
        for i in range(num_inputs):
            # ranking by derivative of the cv
            input_ranks[i] += X.grad[:, i].abs().sum().item()

        print("scv", scv)

# multiply by std dev
if multiply_by_stddev:
    for i in range(num_inputs):
        input_ranks[i] *= in_std[i]

# sort, but don't actually reverse the results
if order_by_importance:

    # print("pre-sort")
    # print("in_num", in_num)
    # print("ranks", input_ranks)
    # print("names", input_names)

    # a[index_array] yields a sorted a
    ranked_indices = input_ranks.argsort()
    input_names = input_names[ranked_indices]
    input_ranks = input_ranks[ranked_indices]
    # print(ranked_indices)

# normalize
input_ranks /= np.sum(input_ranks)

with open(f"{outname}.csv", "w") as csv_file:
    rank_max = np.max(input_ranks)
    rank_list = [
        [i + 1, input_names[-i - 1], round(input_ranks[-i - 1] / rank_max, 3)]
        for i in in_num
    ]
    writer = csv.writer(csv_file)
    writer.writerows(rank_list)

# print("in_num", in_num)
# print("ranks", input_ranks)
# print("names", input_names)

# sys.exit()

# plot
fig = plt.figure(figsize=(5, 0.25 * num_inputs), dpi=100)
ax = fig.add_subplot(111)

if order_by_importance:
    ax.barh(in_num, input_ranks, color="darkgrey", edgecolor="k", linewidth=0.3)
    ax.set_yticklabels(input_names, fontsize=9)
else:
    ax.barh(
        in_num[::-1], input_ranks[::-1], color="darkgrey", edgecolor="k", linewidth=0.3
    )
    ax.set_yticklabels(input_names[::-1], fontsize=9)

ax.set_xlabel("Weight")
ax.set_ylabel("Inputs")
ax.set_yticks(in_num)

# ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
# }}}

# **Download model** {{{

##@title Download model (python binary)

save_hidden_layer_model = True  # @param {type:"boolean"}
save_lda_coeffs = True  # @param {type:"boolean"}
save_pictures = True  # @param {type:"boolean"}
save_checkpoint = True  # @param {type:"boolean"}

print("[Exporting the model]")

# == Create fake input ==
fake_loader = DataLoader(train_labels, batch_size=1, shuffle=False)
fake_input = next(iter(fake_loader))[0].float()

# == Export model ==
# .cuda() required; otherwise
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
# .cpu() definitely does not work
# https://pytorch.org/docs/stable/jit.html#id16
# TODO: CUDA WILL cause problems with plumed
mod = torch.jit.trace(model, fake_input)

mod.save(f"{outname}.pt")
print(f"@@ model exported as: {outname}.pt")

# this file is the bare minimum for metaD
# the following files are nice to have (for publication, etc)

if save_hidden_layer_model:
    # == Export output components ==
    # create a copy of Deep-LDA net with a forward method that returns the hidden variables
    # this model is initialised the same way as the original one
    hidden_model = NN_Hidden(nodes)
    if standardize_inputs:
        hidden_model.set_norm(
            torch.tensor(Mean, dtype=dtype, device=device),
            torch.tensor(Range, dtype=dtype, device=device),
        )

    hidden_model_params = dict(hidden_model.named_parameters())

    # transfer parameters from model to hidden model
    # model has 7 params: lda, nn.{0/2/4}.{weight/bias}, all cuda tensors
    # hidden model is initialised with 6 params (no lda), all cpu tensors
    # these 6 params are replaced (overwritten) by those of the original model

    for name, param in model.named_parameters():
        if name in hidden_model_params:
            # .cuda() does not work here (even if it's redundant)
            # instead, just move the entire model to cuda later
            hidden_model_params[name].data.copy_(param.data)

    # save model

    mod2 = torch.jit.trace(hidden_model, fake_input)
    mod2.save(f"{outname}_hidden.pt")
    print(f"@@ hidden components model saved as: {outname}_hidden.pt")

if save_lda_coeffs:
    # == SAVE LDA COEFFICIENTS ==
    # move back to cpu for writing!
    # 1D (?) array
    # [[0.293259 -0.172498 -0.100701 0.656808 0.665361]]
    with open(f"{outname}_lda.dat", "w") as f:
        f.write(str(model.get_lda().cpu().numpy()))
    print(f"@@ lda coefficients saved in: {outname}_lda.dat")

if save_pictures:
    # == Plot and save results ==
    plot_results(save=True, testing=True)
    print(f"@@ training plots saved as: {outname}_training.png")

if save_checkpoint:
    # == EXPORT CHECKPOINT ==
    # python binary
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        f"{outname}_checkpoint",
    )
    print(f"@@ checkpoint saved as: {outname}_checkpoint")


additional_files = [
    save_hidden_layer_model,
    save_lda_coeffs,
    save_pictures,
    save_checkpoint,
]

# if any(t == True for t in additional_files):
#     # !zip -r {model_name}.zip {model_name}
#     files.download(outname + ".zip")
#     print("- Downloading zip file.")
# else:
#     files.download(tr_folder + outname + ".pt")
#     print("- Downloading model file.")

# }}}

# Use the CV in PLUMED {{{

"""

In order to use the Deep-LDA CV as a collective variable to enhance the sampling, we need to import the trained model inside PLUMED.

To do so, we use Libtorch, which is the Pytorch C++ APIs library. 

Three things are necessary:
- The frozen model exported above (`model.pt`)
- A PLUMED interface to Libtorch, C++ APIs (`PytorchModel.cpp`)
- Download Libtorch and configure PLUMED against it.


**(1) Download Libtorch**

Prebuilt binaries can be downloaded from Pytorch website. Both binaries (cxx11 ABI and pre-cxx11) can be used.

![libtorch download](https://drive.google.com/uc?export=view&id=1nIN4pABqIk7n_xGVgHHR7A6W2Gtk3IZP)

[NOTE 1] Due to the fact that Libtorch is still under development, the PLUMED interface may not work with all versions of Libtorch. This interface has been tested up to version 1.4 of Libtorch, so I recommend downloading the following precompiled version: 
http://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.4.0%2Bcpu.zip 

[NOTE 2] In order for PLUMED to correctly load the serialized model the version of Pytorch and Libtorch should match. By exporting the model with Pytorch 1.4 / 1.5 (you can check in the output of the first cell of the notebook) we can load it with the above version of Libtorch.

**(2) Download PLUMED-LIBTORCH interface**

The interface can be downloaded from [Github](https://github.com/luigibonati/data-driven-CVs/blob/master/code/PytorchModel.cpp).

You need to add this file in the source directory of PLUMED2 (e.g. `plumed2/src/function/`) and, after configuring PLUMED with the Libtorch library, recompile PLUMED. 
Alternatively, this interface can be also loaded in runtime using the LOAD command in the PLUMED input file: 
```
LOAD FILE=PytorchModel.cpp
```
Note that also in this second case you need to configure PLUMED to include the Libtorch library (see below), so I suggest to recompile PLUMED with the .cpp file in it, so that you can immediately detect if the linking was succesful.

**(3) Configure PLUMED and compile it**

If `$LIBTORCH` contains the location of the downloaded binaries, we can configure PLUMED in the following way:

```
./configure  --enable-rpath \
             --disable-external-lapack --disable-external-blas \
             CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
             CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/ -I${LIBTORCH}/include/torch" \
             LDFLAGS="-L${LIBTORCH}/lib -ltorch -lc10 -Wl,-rpath,${LIBTORCH}/lib"
```
NOTE: this command is valid for the pre-cxx11 ABI version. If you downloaded the cxx11 ABI one the corresponding option should be enabled in the configure: 

```
CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
```

**(4) Load the model in the input file**

In the PLUMED input file one should specify the model and the arguments. The interface detects the number of outputs and create a component for each of them, which can be accessed as cv.node-0, cv.node-1, ... 
```
cv: PYTORCH_MODEL MODEL=model.pt ARG=d1,d2,...,dN
```
"""
# }}}

end = timer()
elapsed = end - start
print(f"\nTook {elapsed} s; {distA.shape[1]} CVs, {distA.shape[0]} frames")
