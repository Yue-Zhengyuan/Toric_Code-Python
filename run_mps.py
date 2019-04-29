# 
#   run_mps.py
#   Toric_Code-Python
#   run inverse quasi adiabatic evolution
#
#   created on Apr 24, 2019 by Yue Zhengyuan
#

import os
import sys
import time
import datetime
import para_dict as p
import json
import ast
import lattice as lat
import str_create
import numpy as np
from copy import copy
from str_create import selectRegion

# create result directory
# get path to psi.npy via command line
result_dir = "hz_test/"

outdir = result_dir + 'outfile/'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
parafile = result_dir + 'parameters.txt'

with open(parafile, 'w+') as file:
    pass

python = "~/anaconda3/bin/python"
# python = "python3"

# command parameters
# 1 -> result dir
# 2 -> No. of the string operator
# 3 -> outfile dir

for hz_max in np.linspace(0.1, 0.4, num=4, endpoint=True):
    hz_max = np.around(hz_max, decimals=1)
    command = python + " mps_adiab_evol.py "
    command += result_dir + " " + str(hz_max)
    command += " > " + outdir + "outfile_quasi_" + str(hz_max)
    command += " 2>&1 &"
    os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
