# 
#   para_dict.py
#   Toric_Code-Python
#   initialize parameters
#
#   created on Jan 24, 2019 by Yue Zhengyuan
#

import numpy as np

# system parameters
para = {'nx': 4, 'ny': 4, 
'J': 10.0, 
'hz': 0.0, 'hx': 0.0, 'hy': 0.0, 
'tau': 0.01, 'ttotal': 1}

n = para['nx'] * para['ny']

# Pauli matrices
sx = np.array([[0.,1.], [1.,0.]], dtype=complex) / 2
sy = np.array([[0,-1.0j],[1.0j,0]], dtype=complex) / 2
sz = np.array([[1.,0.], [0.,-1.]], dtype=complex) / 2
iden = np.eye(2, dtype=complex)
