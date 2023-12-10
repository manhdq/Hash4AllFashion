# %%
import os
import os.path as osp
import numpy as np
from utils.metrics import _canonical

# %% Loading some metric results
result_dir = "../results"

for fn in os.listdir(result_dir):
    pre_fn = osp.basename(fn)
    print("=" * 10)
    print(fn)    
    fpath = osp.join(result_dir, fn)
    npzfile = np.load(fpath)
    for f in npzfile:
        print(f, npzfile[f])

# %% [markdown]
# ### Playground
num_scores = 4
num_users = 3

scores = [[[] for _ in range(num_users)] for _ in range(num_scores)]
scores  # [num_scores, num_users]

# %%
arrs = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
]  # [num_scores, B]

for u in range(num_users):
    for n, score in enumerate(arrs):
        for s in score:
            scores[n][u].append(s)

# u = 0
# for n, score in enumerate(arrs):
#     for s in score:
#         scores[n][u].append(s)

scores

# %%
posi, nega = scores[0], scores[1]
posi, nega

# %%
u_labels, u_scores = [], []
for p, n in zip(posi, nega):
    label, score = _canonical(p, n)
    u_labels.append(label)
    u_scores.append(score)

u_scores, u_labels

# %%
