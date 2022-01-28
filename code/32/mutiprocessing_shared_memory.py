from multiprocessing import shared_memory, Process, Manager, Pipe
import time
import numpy as np

pl = []


def s(p, sharr, return_dict):  # send
    data = np.random.random((1, 1024 * 1024, 256)).astype(np.float32)  # 1GB = 1*1024*1024*256*4B
    start_send = time.time()
    sharr[:, :, :] = data[:, :, :]
    p.send('0')
    return_dict['start_send'] = start_send


def r(p, sharr, return_dict):  # recv
    a = np.empty((1, 1024 * 1024, 256), dtype=np.float32)
    p.recv()  # use to sync
    # a[:, :, :] = sharr[:, :, :] # read out
    end_recv = time.time()
    return_dict['end_recv'] = end_recv


manager = Manager()
return_dict = manager.dict()
send_pipe, recv_pipe = Pipe()

res_l = []

shm = shared_memory.SharedMemory(create=True, size=1073741824)
sharr = np.ndarray((1, 1024 * 1024, 256), dtype=np.float32, buffer=shm.buf)

for _ in range(3):
    p1 = Process(target=r, args=(recv_pipe, sharr, return_dict,))
    p2 = Process(target=s, args=(send_pipe, sharr, return_dict,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    res_l.append(return_dict['end_recv'] - return_dict['start_send'])

del sharr
shm.close()
shm.unlink()
print(sum(res_l) / 3)

# no read out
# 0.25899219512939453

# read out
# 0.46038611729939777
