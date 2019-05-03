# 
#   main_mps.py
#   Toric_Code-Python
#   apply the Trotter gates to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import mps, mpo, gnd_state
import para_dict as p
import lattice as lat
import sys, os, datetime, time, json, ast
from copy import copy
from tqdm import tqdm

args = copy(p.args)
nx = 7
ny = args['ny']
sep = 10
hz = args['hz']

result_dir = "mpopair_quasi_2019-05-02_20-45/"
label = "quasi"
op = np.load(result_dir + '{}_op_{}by{}_sep-{}_hz-{:.2f}.npy'.format(label, nx, ny, sep, hz))
op = list(op)
entg_file = result_dir + '/entg_{}_{}by{}_{}.txt'.format(label, nx, ny, sep)
with open(entg_file, 'w+') as file:
    pass

for site in tqdm(range(len(op))):
    op, entg = mpo.position(op, site, args, oldcenter=site - 1, compute_entg=True)
    y = int(site / (nx - 1))
    x = site % (nx - 1)
    if y % 2 == 0:
        bonddir = 'r'
    else:
        bonddir = 'd'
    with open(entg_file, 'a+') as file:
        file.write("{}\t{}\t{}\t{}\t{}\n".format(site, x, int(y/2), bonddir, entg))
