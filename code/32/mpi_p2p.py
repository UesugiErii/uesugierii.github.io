from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:

    data = np.zeros((1, 1024 * 1024, 256), dtype=np.float32)
    time.sleep(3)  # make sure that rank=1 had allocated space to receive the array
    print(time.time())
    comm.Send(data, dest=1)

elif rank == 1:

    data = np.empty((1, 1024 * 1024, 256), dtype=np.float32)
    comm.Recv(data, source=0)
    print(time.time())

# mpirun -n 2 python3 mpi_p2p.py
# 0.1895733674367269
