from multiprocessing import Process, Array, Manager, Event
from ctypes import c_float
import numpy as np
import time


def s(shm, e, return_dict):  # send
    data = np.random.random((1, 1024 * 1024, 256)).astype(np.float32)
    start_send = time.time()
    shm[:, :, :] = data[:, :, :]
    e.set()
    return_dict['start_send'] = start_send


def r(shm, e, return_dict):  # recv
    data = np.empty((1, 1024 * 1024, 256)).astype(np.float32)
    e.wait()
    data[:, :, :] = shm[:, :, :]  # read out
    end_recv = time.time()
    e.clear()
    return_dict['end_recv'] = end_recv


manager = Manager()
return_dict = manager.dict()
e = Event()
e.clear()

arr = Array(c_float, np.empty((1024 * 1024 * 256,), dtype=np.float32))  # 1 GB
shm = np.frombuffer(arr.get_obj(), dtype=np.float32)
shm.resize((1, 1024 * 1024, 256))

pl = []
res_l = []

for _ in range(3):
    p1 = Process(target=r, args=(shm, e, return_dict,))
    p2 = Process(target=s, args=(shm, e, return_dict,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    res_l.append(return_dict['end_recv'] - return_dict['start_send'])

print(sum(res_l) / 3)

# no read out
# 0.39174636205037433

# read out
# 0.5768861770629883
