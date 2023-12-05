# %%
import torch
import numpy as np

from collections import defaultdict
from icecream import ic

# %%
a = defaultdict(tuple)

def get_val(d):
    print(type(d))
    ic(d["a"])

get_val(a)    

# %%
x = torch.randn((1, 512))
y = torch.randn((1, 512))

xs = [x]
xs = torch.cat(xs, 0)
print(xs.shape)

# %%
results = dict(np.load("results/fhn_voe_t3_12_1.npz"))
print(results)

# %%
a = np.arange(4*5).reshape(4, 5)
a

# %%
iu1, iu2 = np.triu_indices(4, k=1)
a[iu1], a[iu2]

# %%
b = np.arange(4).reshape(1, 4)
b

# %%
a * b


# %%
class Meter(object):
    def __init__(self):
        self.val = 0

    def update(self):
        self.val += 1

    @property
    def cur(self):
        return self.val


sample = Meter()
sample.update()
sample.cur


# %%
def foo():
    return ((1, 2, 3), (4, 5, 6))


a, b, c, d, e, f = foo()

# %%
a = {"1":2, "3":4, "2":3}

def do_st(**inputs):
    print(inputs["1"])

do_st(**a)

# %%
