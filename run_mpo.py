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
import str_create as crt
import numpy as np
from copy import copy
from str_create import selectRegion

# create result directory
# get path to psi.npy via command line
result_dir = sys.argv[1]
# result_dir = "result_mps_2019_4_24_18_31/"
os.makedirs(result_dir, exist_ok=True)

str_sep_list = [6, 10, 14]
sys_size_list = range(3,8)

with open(result_dir + '/parameters.txt', 'w+') as file:
    pass

python = "~/anaconda3/bin/python"
# python = "python3"
# create string list (can handle both x-PBC and OBC)
for nx in sys_size_list:
    for str_sep in str_sep_list:
        # command parameters
        # 1 -> result dir
        # 2 -> system size nx
        # 3 -> string separation
        # 4 -> outfile dir
        command = python + " mpo_adiab_evol.py "
        command += result_dir + " " + str(nx) + ' ' + str(str_sep)
        command += " > " + result_dir + "outfile_" + str(nx) + '_' + str(str_sep)
        command += " 2>&1 &"
        os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
