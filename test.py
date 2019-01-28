import numpy as np
from itertools import product
import para_dict as p
import gates
import copy

# swap gate test
gate = np.zeros((2,2,2,2), dtype=complex)
for a,b,c,d in product(range(2), repeat=4):
    if (a == d and b == c):
        gate[a,b,c,d] = 1.0
gate = np.einsum('abcd->acbd', gate)
gate_dag = np.einsum('abcd->cdab', gate)

str_op = []
for i in range(2):
    str_op.append(np.zeros((1,1,2,2), dtype=complex))
str_op[0][0,0,:,:] = np.array([[1,2],[3,4]])
str_op[1][0,0,:,:] = np.array([[5,6],[7,8]])

result = np.einsum('ijab,jkcd,efac,bdgh->ikegfh',str_op[0],str_op[1],gate,gate_dag)
result = np.reshape(result,(4,4))
u,s,v = np.linalg.svd(result)

u = np.reshape(u[:,0], (1,1,2,2)) * np.sqrt(s[0])
v = np.reshape(v[0,:], (1,1,2,2)) * np.sqrt(s[0])
print("Hello world!")