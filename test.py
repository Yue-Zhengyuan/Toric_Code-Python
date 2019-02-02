import numpy as np
from itertools import product
import para_dict as p
import gates
import gateTEvol
import copy

# svd test
mat = np.array([[1,0,0,1]])
u,s,v = np.linalg.svd(mat)

print("Hello world!")