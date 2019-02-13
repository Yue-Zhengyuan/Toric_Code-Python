# 
#   main.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import mps
import mpo
import sys
import copy
import time
import datetime
import os
import json

# as requested in comment
exDict = {'exDict': 1}

with open('file.txt', 'w') as file:
     file.write(json.dumps(exDict)) # use `json.loads` to do the reverse