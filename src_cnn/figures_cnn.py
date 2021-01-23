import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle

from train_v import generate_episodes
from cube_model_naive import Cube as env
from cnn import conv_net

mpl.style.use('seaborn')
np.random.seed(23)

# GLobal variables
_, cnn_apply = conv_net()
N = 1024
K = 26
T = (N, K)
S = (N, K) + env.terminal_state.shape
PARAMS_PATH = '../CNN_params'


if os.path.exists('../pickles/test_set'):
    with open('../pickles/test_set', mode='rb') as f:
        TEST_SET = pickle.load(f)
        print('Loaded test_set...')
else:
    TEST_SET = generate_episodes(env, N, K)[0]
    # Save TEST_SET
    with open('../test_set.npy', mode='wb') as f:
        pickle.dump(TEST_SET, f)

# Read CNN parameters from each epoch
param_list = [f for f in os.listdir(PARAMS_PATH) if f.endswith('.npy')]
param_list.sort(key=lambda x: int(x[11:-4]))


# ===--------------------- State Values trough Epochs ---------------------=== #

if os.path.exists('../pickles/epoch_values'):
    with open('../pickles/epoch_values', mode='rb') as f:
        EPOCH_VALUES = pickle.load(f)
        print('Loaded epoch_values...')
else:
    EPOCH_VALUES = []
    for p in param_list:
        print('Evaluating ', p)
        params = jnp.load(os.path.join(PARAMS_PATH, p), allow_pickle=True)
        V = []
        for i in range(0, N * K, 512):
            v = cnn_apply(params, TEST_SET[i:i+512])
            V.append(v.ravel())
        V = np.hstack(V).reshape(N, K)
        EPOCH_VALUES.append(V)

    # Save EPOCH_VALUES
    with open('../epoch_values.npy', mode='wb') as f:
        pickle.dump(EPOCH_VALUES, f)


# MEANS = []
# STDS = []
# for arr in EPOCH_VALUES:
#     MEANS.append(np.mean(arr, axis=0))
#     STDS.append(np.std(arr, axis=0))
# MEANS = np.vstack(MEANS)
# STDS = np.vstack(STDS)

# fig, ax = plt.subplots()
# sns.heatmap(MEANS.T)


# ===---------------------- Best First Search vs A* ----------------------=== #

params = jnp.load(os.path.join(PARAMS_PATH, param_list[-1]), allow_pickle=True)
episodes = TEST_SET.reshape(S)

# SOLUTIONS = {}
# for i in range(K):
#     cubes = episodes[:, i]
#     solved = 0
#     for state in cubes:
#         n = 0
#         cube = env()
#         cube.state = state.copy()
#         while n < 30:
#             n += 1
#             if cube.is_solved():
#                 solved += 1
#                 break
#             children, rewards = env.expand_state(cube._state)
#             vals = cnn_apply(params, children).ravel()
#             a = int(np.argmax(vals + rewards.ravel()))
#             cube.step(a)
#     print('Distance {} solve rate: {:.2f}'.format(i, solved / N))
#     SOLUTIONS[i] = solved


# with open('../bfs_solutions', mode='wb') as f:
#     pickle.dump(SOLUTIONS, f)

import time

A_SOLUTIONS = {}
A_SOLUTIONS_MOVES = {}
A_TIMES = {}
FAILURES = []
os.chdir('..')
from Astar import search, heuristic_from_nn
H = heuristic_from_nn(cnn_apply, params)
print('Starting A*...')
for i in range(15):
    cubes = episodes[:, i]
    solved = 0
    x = env()
    solved = 0
    A_SOLUTIONS_MOVES[i] = []
    A_TIMES[i] = []
    for state in cubes[::10]:
        x.state = state
        tic = time.time()
        solution = search(x, H, 1000)
        toc = time.time()
        A_TIMES[i].append(toc - tic)
        A_SOLUTIONS_MOVES[i].append(solution)
        if solution or env.is_terminal(state):
            solved += 1
        else:
            FAILURES.append(state)
    print('Distance {} solve rate: {:.2f}'.format(i, solved / 103))
    A_SOLUTIONS[i] = solved