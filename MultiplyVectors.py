import time
import numba as cuda
import numpy as np
import matplotlib as mpl

# We still require the cupy package which is only usable with cuda enabled GPUs with a compute capability over 3.0 (Rtx 3000 series are 7.5 )

# TUTORIALS
# https://www.youtube.com/watch?v=wa0EmEq5Otw
# https://www.youtube.com/watch?v=r9IqwpMR9TE
# https://www.youtube.com/watch?v=dPQnFXD7DxM

#A and B are the input vectors, returns the mutiplied vector
@cuda.jit("float32[:](float32[:],float32[:])",nopython=True)
def VectorMultiplication(a,b):
    return a*b

def main():

    MATRIX_SIZE = 1024*10240 #size of the declared arrays

    A = np.ones(MATRIX_SIZE, dtype=np.float32)
    B = np.ones(MATRIX_SIZE, dtype=np.float32)
    C = np.zeros(MATRIX_SIZE, dtype=np.float32)

    start = time.time()

    C = VectorMultiplication(A,B)
    #DO CUDA

    executionTime = time.time() - start

    print("C[:6] =",C[:6])
    print("This vector multiplication took %f secconds" %executionTime)


main()

