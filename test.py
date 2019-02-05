import numpy as np
from itertools import product
import para_dict as p
import gates
import gateTEvol
import copy

# hamiltonian test
para = p.para
for i in np.arange(para['nx'] + 1, p.n - 3 * para['nx'] + 2, 2 * para['nx'] - 1, dtype=int):
    for j in np.arange(0, para['nx'] - 2, 1, dtype=int):
        u = i + j
        l = u + para['nx'] - 1
        r = l + 1
        d = l + para['nx']
        sites = [u, l, r, d]
        print(sites)

print("Hello world!")