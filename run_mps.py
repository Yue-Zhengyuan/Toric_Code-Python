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
result_dir = sys.argv[1]

result_dir += "/quasi"
outdir = result_dir + '/outfile'
os.makedirs(result_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
parafile = result_dir + '/parameters.txt'
resultfile = result_dir + '/dressed_result.txt'

# get the number of strings
# create string list (can handle both x-PBC and OBC)
str_list = str_create.str_create(args, args['ny'] - 1)
# used_str = [-1]
used_str = range(len(str_list))

# save parameters
with open(parafile, 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
    file.write('\n\nUsing String Operators:')
    file.write('\n\n')
    for i in used_str:
        bond_on_str, area, circum = str_create.convertToStrOp(str_list[i], args)
        bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
        region = selectRegion(bond_on_str, 1, args)
        bond_list.sort()
        region.sort()
        # Output information
        file.write(str(area) + '\t' + str(circum) + '\n')
        file.write("bond on str: \n" + str(bond_on_str) + '\n')
        file.write("bond number: \n" + str(bond_list) + '\n')
        file.write("bond within distance 1: \n" + str(region) + '\n')

python = "/anaconda3/bin/python"
# python = "python3"

# command parameters
# 1 -> result dir
# 2 -> No. of the string operator
# 3 -> outfile dir

for i in used_str:
    command = python + " mps_quasi_evol.py "
    command += result_dir + " " + str(i)
    command += " > " + outdir + "/outfile_quasi_" + str(i)
    command += " 2>&1 &"
    os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
