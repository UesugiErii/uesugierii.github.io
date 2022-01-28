from multiprocessing import Process, Array, Manager
from ctypes import c_float
import numpy as np

arr = Array(c_float, np.empty((1024 * 1024 * 128,), dtype=np.float32))

t = np.empty((1024 * 1024 * 128,)).astype(np.float32)  # 512MB

#                    c_float               c_int
t[:] = arr[:]  # 7.312359650929769    9.989207824071249  Will occupy 4G memory during the assignment
arr[:] = t[:]  # 13.619366963704428   17.98023788134257

# a = np.array(arr)  # 386.65015149116516  

# why 4GB memory?
# The assignment process needs to go through python intermediate processing

# In [2]: sys.getsizeof(np.zeros((128,))[0])
# Out[2]: 32

# A floating point number becomes 8 times bigger (4B -> 32B)
# so is c_int

# Do not use this method to exchange data between different processes
