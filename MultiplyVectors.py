import time
import numba as cuda
import numpy as np
import matplotlib as mpl
# We still require the cupy package which is only usable with cuda enabled GPUs with a compute capability over 3.0 (Rtx 3000 series are 7.5 )
# https://www.youtube.com/watch?v=wa0EmEq5Otw Introduction tutorial to cuda
# https://www.youtube.com/watch?v=r9IqwpMR9TE



def main():
    N = 6400000 #size of the declared arrays

    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    C = np.ones(N,dtype=np.float32)

    start = time.time()
    #MultiplyMyVector(A, B, C)

    executionTime = time.time() - start

    print("C[:6] = ", C[:6])
    print("This vector multiplication took %f secconds" %executionTime)


main()

