# 
#   main.py
#   Toric_Code-Python
#   apply the gates to string operator
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import lattice as lt
import gates as gt
import para_dict as p

# create string operator MPO
# labelling: [site, L vir leg, R vir leg, U phys leg, D phys leg]
str_op = np.zeros((p.n,1,1,2,2), dtype=complex)
sites_on_str = [12, 13, 14, 15]
for i in range(p.n):
    if i in sites_on_str:
        str_op[i,:,:,:,:] = p.sz
    else:
        str_op[i,:,:,:,:] = p.iden

# generate gates for one step of time evolution


# apply gates to the string operator MPO
print('Hello world')