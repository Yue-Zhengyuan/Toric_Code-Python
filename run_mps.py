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

args = copy(p.args)
# create result directory
# get system time
nowtime = datetime.datetime.now()
result_dir = '_'.join(['result_mpo', str(nowtime.year), str(nowtime.month), 
str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)

outdir = result_dir + '/outfile'
os.makedirs(outdir, exist_ok=True)

# get the number of strings
# create string list (can handle both x-PBC and OBC)
str_list = str_create.str_create(args, args['ny'] - 1)

# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
    file.write('\n\nUsing String Operators:')
    file.write('\n\n')
    # for i in range(len(line)):
    for i in np.arange(0, len(str_list), 2, dtype=int):
        bond_on_str, str_area, circum = str_create.convertToStrOp(str_list[i], args)
        file.write(str(str_area) + '\t' + str(bond_on_str) + '\n')

# command parameters
# 1 -> result dir
# 2 -> No. of the string operator
# 3 -> outfile dir
# for i in range(len(line)):
python = "/anaconda3/bin/python"

for i in np.arange(0, len(str_list), 2, dtype=int):
    command = python + " main_mpo_quasi.py "
    command += result_dir + " " + str(i) + " > "
    command += outdir + "/outfile_quasi" + str(i)
    command += " 2>&1 &"
    os.system(command)
    command = python + " main_mpo_adiab.py "
    command += result_dir + " " + str(i) + " > "
    command += outdir + "/outfile_adiab" + str(i)
    command += " 2>&1 &"
    os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
