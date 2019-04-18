# 
#   para_dict.py
#   Toric_Code-Python
#   TEBD / tMPO parameters
#
#   created on Jan 24, 2019 by Yue Zhengyuan
#

import numpy as np

# system parameters
# current convention:
# for the same system width (x)
# x_PBC = x_OBC - 1 (will be fixed in the future if time allows)
args = {'nx': 5, 'ny': 20, 
'U': 0, 'g': 1.0, 
'hz': 0.4, 
'xperiodic': True,
'tau': 0.005, 'ttotal': 1.0, 
'cutoff': 1.0E-8, 'bondm': 512}

n = 2 * (args['nx'] - 1) * args['ny']
# Y-non-periodic
n -= args['nx'] - 1
# X-non-periodic
n += args['ny'] - 1
args.setdefault('n', n)
# n in case of periodic X
if args['xperiodic'] == True:
    args.setdefault('real_n', n - (args['ny'] - 1))
else:
    args.setdefault('real_n', n)

# Pauli matrices
sx = np.array([[ 0,   1], [  1,  0]], dtype=complex)
sy = np.array([[ 0, -1j], [ 1j,  0]], dtype=complex)
sz = np.array([[ 1,   0], [  0, -1]], dtype=complex)
iden = np.eye(2, dtype=complex)
