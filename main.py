# 
#   main.py
#   Toric_Code-Python
#   apply the gates to string operator
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import gateTEvol as evol

# index order convention
# 
# Tensor
#       a      c
#      _|_    _|_
#  i --| |----| |--j
#      -|-    -|-
#       b      d
#
#  index order: iabcdj
# 
# Gate
#       a      c
#      _|_    _|_
#      | |----| |
#      -|-    -|-
#       b      d
#
#  index order: abcd

# create string operator MPO
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
sites_on_str = [12, 13, 14, 15]
for i in range(p.n):
    if i in sites_on_str:
        str_op[i][0,:,:,0] = p.sz
    else:
        str_op[i][0,:,:,0] = p.iden

# generate gates for one step of time evolution
gateList = gates.makeGateList(str_op, p.para)
# apply gates to the string operator MPO
evol.gateTEvol(str_op, gateList, 1.0, p.para['tau'])

print('Hello world')