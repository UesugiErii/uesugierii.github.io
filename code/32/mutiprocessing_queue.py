from multiprocessing import Process, Queue, Manager
import time
import numpy as np

pl = []


def s(q, return_dict):  # send
    data = np.random.random((1, 1024 * 1024, 256)).astype(np.float32)  # 1GB = 1*1024*1024*256*4B
    start_send = time.time()
    q.put(data)
    return_dict['start_send'] = start_send


def r(q, return_dict):  # recv
    a = q.get()
    end_recv = time.time()
    return_dict['end_recv'] = end_recv


manager = Manager()
return_dict = manager.dict()

q = Queue()

res_l = []

for _ in range(3):
    p1 = Process(target=r, args=(q, return_dict,))
    p2 = Process(target=s, args=(q, return_dict,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    res_l.append(return_dict['end_recv'] - return_dict['start_send'])

print(sum(res_l) / 3)

# 1.9592033227284749
