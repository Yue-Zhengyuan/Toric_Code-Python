# 
#   para_dict.py
#   Toric_Code-Python
#   initialize parameters
#
#   created on Jan 24, 2019 by Yue Zhengyuan
#

import numpy as np

# system parameters
para = {'nx': 10, 'ny': 10, 'v_U': 10.0,
 'p_g': 1.0, 'hz': 0.0, 'tau': 0.01}
n = 2 * (para['nx'] - 1) * para['ny']
# Y-boundary
n -= para['nx'] - 1
# X-non-periodic
n += para['ny'] - 1

# Pauli matrices
sx = np.array([[0.,1.], [1.,0.]], dtype=complex) / 2
sz = np.array([[1.,0.], [0.,-1.]], dtype=complex) / 2
iden = np.eye(2, dtype=complex)
