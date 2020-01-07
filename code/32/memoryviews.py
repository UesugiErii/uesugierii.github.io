# modified from
# https://python3-cookbook.readthedocs.io/zh_CN/latest/c11/p13_sending_receiving_large_arrays.html

from multiprocessing import Process, Manager
import time
import numpy as np
from socket import *

pl = []

port = 25005


# s->c

def s(return_dict):  # send server

    arr = np.random.random((1, 1024 * 1024, 256)).astype(np.float32)
    #           s<->s       TCP
    s = socket(AF_INET, SOCK_STREAM)
    s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen(1)
    c, a = s.accept()

    dest = c
    # arr = arr

    start_send = time.time()

    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

    return_dict['start_send'] = start_send


def r(return_dict):  # recv client
    arr = np.zeros((1, 1024 * 1024, 256), dtype=np.float32)
    time.sleep(5)  # make sure that server is started
    c = socket(AF_INET, SOCK_STREAM)
    c.connect(('localhost', port))

    source = c
    # arr = arr

    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

    end_recv = time.time()
    return_dict['end_recv'] = end_recv


manager = Manager()
return_dict = manager.dict()

res_l = []

for _ in range(3):
    p1 = Process(target=s, args=(return_dict,))
    p2 = Process(target=r, args=(return_dict,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    res_l.append(return_dict['end_recv'] - return_dict['start_send'])

print(sum(res_l) / 3)

# 0.34012564023335773
