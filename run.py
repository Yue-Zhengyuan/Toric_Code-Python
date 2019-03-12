import os
import sys
import time
import datetime
import para_dict as p
import json
import ast

# create result directory
# get system time
nowtime = datetime.datetime.now()
result_dir = '_'.join(['result_mpo', str(nowtime.year), str(nowtime.month), 
str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)

outdir = result_dir + '/outfile'
os.makedirs(outdir, exist_ok=True)

# get the number of strings
with open('list.txt', 'r') as f:
    line = f.readlines()

# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
    file.write('\n\nUsing String Operators:')
    file.write('\n\n')
    for i in range(len(line)):
        bond_on_str = ast.literal_eval(line[i])[0:-1]
        file.write(str(bond_on_str) + '\n')    # bonds

# command parameters
# 1 -> result dir
# 2 -> No. of the string operator
# 3 -> outfile dir
for i in range(len(line)):
    command = "python main_mpo_quasi.py "
    command += result_dir + " " + str(i) + " > "
    command += outdir + "/outfile_quasi" + str(i)
    command += " 2>&1 &"
    os.system(command)
    command = "python main_mpo_adiab.py "
    command += result_dir + " " + str(i) + " > "
    command += outdir + "/outfile_adiab" + str(i)
    command += " 2>&1 &"
    os.system(command)

# if no error occurs, remove the outfiles
# python main_mpo_quasi.py > outfile_mpoquasi 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
