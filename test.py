import numpy as np
from itertools import product
import para_dict as p

para = p.para

for i in np.arange(para['nx'] + 1, p.n - para['nx'] + 2, 2 * para['nx'] - 1, dtype=int):
    for j in np.arange(0, para['nx'] - 2, 1, dtype=int):
        u = i + j
        l = u + para['nx'] - 1
        r = l + 1
        d = l + para['nx']
        if (d > p.n):
            current_sites = [u, l, r, d - p.n]
        else:
            current_sites = [u, l, r, d]
        print(current_sites)