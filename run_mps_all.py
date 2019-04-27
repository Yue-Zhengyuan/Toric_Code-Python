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

args = copy(p.args)
# create result directory
# get path to psi.npy via command line
result_dir = "result_PBC/x_by_20/"

outdir = result_dir + 'outfile/'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
parafile = result_dir + 'parameters.txt'
resultfile = result_dir + 'dressed_result.txt'

python = "~/anaconda3/bin/python"
# python = "python3"

# command parameters
# 1 -> result dir
# 2 -> No. of the string operator
# 3 -> outfile dir

for i in range(4,8):
    command = python + " mps_quasi_evol_all.py "
    command += result_dir + " " + str(i)
    command += " > " + outdir + "outfile_quasi_" + str(i)
    command += " 2>&1 &"
    os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
