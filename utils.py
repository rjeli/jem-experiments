import time
import contextlib

import torch

@contextlib.contextmanager
def timer(msg):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f'ms to {msg}: {(t1-t0)*1000}')

def energy_of_pred(pred):
    return -torch.logsumexp(pred, dim=1)
