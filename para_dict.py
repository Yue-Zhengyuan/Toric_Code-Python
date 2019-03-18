# 
#   para_dict.py
#   Toric_Code-Python
#   initialize parameters
#
#   created on Jan 24, 2019 by Yue Zhengyuan
#

import numpy as np

# system parameters
args = {'nx': 7, 'ny': 7, 
'U': 0.0, 'g': 1.0, 
'hz': 0.2, 'hx': 0.0, 'hy': 0.0, 
'tau': 0.01, 'ttotal': 1.0, 
'cutoff': 1.0E-6, 'bondm': 256, 'scale': False}

n = 2 * (args['nx'] - 1) * args['ny']
# Y-boundary
n -= args['nx'] - 1
# X-non-periodic
n += args['ny'] - 1
args.setdefault('n', n)

# Pauli matrices
sx = np.array([[0.,1.], [1.,0.]], dtype=complex)
sy = np.array([[0,-1.0j],[1.0j,0]], dtype=complex)
sz = np.array([[1.,0.], [0.,-1.]], dtype=complex)
iden = np.eye(2, dtype=complex)
