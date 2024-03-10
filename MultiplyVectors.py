import  numpy as np
import time  
from numba import vectorize

#@vectorized(["outputType(inputtype, inputtype)"] target = "cpu", or "cuda", )
@vectorize(["float32(float32, float32)"], target = 'cuda')
def MultiplyMyVec(a,b):
    return a*b
        
def main():
    N  = 64000000 # size per declared array 
    
    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)
    C = np.ones(N, dtype = np.float32)
    
    start = time.time() # start my timer 
    C = MultiplyMyVec(A,B)
    vectormultiply_time = time.time() - start
    
    print("C[:6]:", C[:6])
    print("C[:6]:", C[-6:])
    print("This multiplication  took %.3f seconds" % vectormultiply_time)
    
    
main()