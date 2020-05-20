from multiprocessing import Process, Pipe, Manager
import time
import numpy as np

pl = []


def s(p, return_dict):  # send
    # Send data size range -2147483648 <= number <= 2147483647
    # can't send 2GB data
    data = np.random.random((1, 1024 * 1024, 256)).astype(np.float32)  # 1GB = 1*1024*1024*256*4B
    start_send = time.time()
    p.send(data)
    return_dict['start_send'] = start_send


def r(p, return_dict):  # recv
    a = p.recv()
    end_recv = time.time()
    return_dict['end_recv'] = end_recv


manager = Manager()
return_dict = manager.dict()

send_pipe, recv_pipe = Pipe()

res_l = []

for _ in range(3):
    p1 = Process(target=r, args=(recv_pipe, return_dict,))
    p2 = Process(target=s, args=(send_pipe, return_dict,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    res_l.append(return_dict['end_recv'] - return_dict['start_send'])

print(sum(res_l) / 3)

# 1.7048665682474773
