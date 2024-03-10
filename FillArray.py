
import numpy as np
import  time 
from numba import cuda, jit

# This func will run on CPU
def FillArrayWithoutGPU(a):
    for i in range(10000000):
        a[i] += 1
        
# This func will run on a GPU
@jit(target_backend = 'cuda')
def FillArrayWithGPU(a):
    for i in  range(10000000):
        a[i] += 1
        
        
#Main
a = np.ones(10000000, dtype = np.float64)
t0 = time.time()
FillArrayWithoutGPU(a)
print("On  a CPU:", time.time() - t0)


t0 = time.time()
FillArrayWithGPU(a)
print("On  a GPU:", time.time() - t0)