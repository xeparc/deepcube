import numpy as np
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle

from train_v import generate_episodes
from rubick import RubickCube as env
from fcnn import value_appoximator

mpl.style.use('seaborn')
np.random.seed(23)

# GLobal variables
_, apply = value_appoximator()
N = 1024
K = 26
T = (N, K)
S = (N, K) + env.terminal_state.shape
PARAMS_PATH = '../pickles/FCNN_params/all'
TEST_SET = generate_episodes(env, N, K)[0]

# Read CNN parameters from each epoch
param_list = [f for f in os.listdir(PARAMS_PATH) if f.endswith('.npy')]
param_list.sort(key=lambda x: int(x[6:-4]))


# ===--------------------- State Values trough Epochs ---------------------=== #

EPOCH_VALUES = []
for p in param_list:
    print('Evaluating ', p)
    params = jnp.load(os.path.join(PARAMS_PATH, p), allow_pickle=True)
    V = []
    for i in range(0, N * K, 512):
        v = apply(params, TEST_SET[i:i+512])
        V.append(v.ravel())
    V = np.hstack(V).reshape(N, K)
    EPOCH_VALUES.append(V)

# Save TEST_SET
with open('../pickles/fcnn_epoch_values', mode='wb') as f:
    pickle.dump(EPOCH_VALUES, f)

# Save EPOCH_VALUES
with open('../pickles/fcnn_test_set', mode='wb') as f:
    pickle.dump(TEST_SET, f)


MEANS = []
STDS = []
for arr in EPOCH_VALUES:
    MEANS.append(np.mean(arr, axis=0))
    STDS.append(np.std(arr, axis=0))
MEANS = np.vstack(MEANS)
STDS = np.vstack(STDS)

fig, ax = plt.subplots()
sns.heatmap(MEANS.T)


# ===---------------------- Best First Search vs A* ----------------------=== #

params = jnp.load(os.path.join(PARAMS_PATH, param_list[-1]), allow_pickle=True)
episodes = TEST_SET.reshape(S)

SOLUTIONS = {}
for i in range(K):
    cubes = episodes[:, i]
    solved = 0
    for state in cubes:
        n = 0
        cube = env()
        cube.state = state.copy()
        while n < 30:
            n += 1
            if cube.is_solved():
                solved += 1
                break
            children, rewards = env.expand_state(cube.state)
            vals = apply(params, children).ravel()
            a = int(np.argmax(vals + rewards.ravel()))
            cube.step(a)
    print('Distance {} solve rate: {:.2f}'.format(i, solved / N))
    SOLUTIONS[i] = solved


with open('../pickles/fcnn_bfs_solutions', mode='wb') as f:
    pickle.dump(SOLUTIONS, f)


A_SOLUTIONS = {}
A_SOLUTIONS_MOVES = {}
FAILURES = []
os.chdir('..')
from Astar import search, heuristic_from_nn
H = heuristic_from_nn(apply, params)

for i in range(2):
    cubes = episodes[:, i]
    solved = 0
    x = env()
    solved = 0
    A_SOLUTIONS_MOVES[i] = []
    for state in cubes:
        x.state = state
        solution = search(x, H, 1000)
        A_SOLUTIONS_MOVES[i].append(solution)
        if solution:
            solved += 1
        else:
            FAILURES.append(state)
    print('Distance {} solve rate: {:.2f}'.format(i, solved / N))
    A_SOLUTIONS[i] = solved
