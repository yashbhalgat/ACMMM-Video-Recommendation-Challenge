from __future__ import division
from numba import cuda, float32
import numpy
from numpy import genfromtxt
import math
import pdb

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"
shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

def load_set(shows_dir, movies_dir, phase="test"):
    # loading test set
    shows_set = genfromtxt(shows_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
    shows_set = list(shows_set)
    shows_set = [int(i) for i in shows_set]
    movies_set = genfromtxt(movies_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
    movies_set = list(movies_set)
    movies_set = [int(i) for i in movies_set]

    return shows_set, movies_set


@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

if __name__ == '__main__':
    shows_valid_set, movies_valid_set = load_set(shows_dir, movies_dir, "val")
    K = numpy.load("kernel_stats/kernel_LSTM_stats.npy")
    triplets = numpy.load("kernel_stats/triplets.npy")
    tau = numpy.load("kernel_stats/tau_50000.npy")

    triplets = triplets[0:50001]

    indnonzero = numpy.where(tau!=0)[0]
    # indnonzero = indnonzero[8:]    # To make it a multiple of 16
    triplets = triplets[indnonzero,:]
    tau = tau[indnonzero]

    T = len(triplets)

    pl_inds = [triplets[t][0] for t in range(T)]
    plplus_inds = [triplets[t][1] for t in range(T)]
    plminus_inds = [triplets[t][2] for t in range(T)]
    q_vids = [i for i in range(7536)]

    pdb.set_trace()

    score = K[shows_valid_set, :][:, q_vids]

    num_blocks = 8
    for i in range(num_blocks):
        TT = T//8
        pl_inds_iter = pl_inds[i*TT:(i+1)*TT]
        plplus_inds_iter = plplus_inds[i*TT:(i+1)*TT]
        plminus_inds_iter = plminus_inds[i*TT:(i+1)*TT]
        tau_iter = tau[i*TT:(i+1)*TT]

        K1 = K[shows_valid_set, :][:, pl_inds_iter]
        TAU = numpy.diag(tau_iter)
        K2 = K[plplus_inds_iter, :][:, q_vids] - K[plminus_inds_iter, :][:, q_vids]

        # pdb.set_trace()

        # # The data array
        # A = numpy.full((TPB*2, TPB*3), 3, numpy.float) # [32 x 48] matrix containing all 3's
        # B = numpy.full((TPB*3, TPB*1), 4, numpy.float) # [48 x 16] matrix containing all 4's

        A_global_mem = cuda.to_device(K1)
        B_global_mem = cuda.to_device(TAU)
        C_global_mem = cuda.device_array((K1.shape[0], TAU.shape[1])) # [32 x 16] matrix result

        # Configure the blocks
        threadsperblock = (TPB, TPB)
        blockspergrid_x = int(math.ceil(K1.shape[0] / threadsperblock[1]))
        blockspergrid_y = int(math.ceil(TAU.shape[1] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Start the kernel 
        fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        res = C_global_mem.copy_to_host()

        # pdb.set_trace()

        A_global_mem = cuda.to_device(res)
        B_global_mem = cuda.to_device(K2)
        C_global_mem = cuda.device_array((res.shape[0], K2.shape[1])) # [32 x 16] matrix result

        # Configure the blocks
        threadsperblock = (TPB, TPB)
        blockspergrid_x = int(math.ceil(res.shape[0] / threadsperblock[1]))
        blockspergrid_y = int(math.ceil(K2.shape[1] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Start the kernel 
        fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
        product = C_global_mem.copy_to_host()

        score = score + product
        print(str(i), "/", str(T), "done")

    pdb.set_trace()
    numpy.save("kernel_stats/score_kernel_stats_"+str(T)+".npy", score)

    aaaa=4
