import time
import numpy as np
import jax
import jax.numpy as jnp
import os
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from functools import partial
from math import ceil

from fcnn import value_appoximator as model_fn

#-------------------- data generation utilities --------------------#
def expand_states(states, env):
    """ Given an array of states use the model of the environment to
    obtain the descendants of these states and their respective rewards.
    Return the descendants and the rewards.

    @param states (Array[state]): A numpy array of valid states of the environment.
                                  The shape of the array is (N, state.shape),
                                  where N is the number of states.
    @param env (Cube Object): A Cube object representing the environment.
    @returns children (Array[state]): A numpy array giving the children of the input states
                                      The shape of the array is (N * num_acts, state.shape).
    @returns rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                              respective rewards.
    """
    zipped = (env.expand_state(s) for s in states)
    children, rewards = list(zip(*zipped))
    children = np.vstack(children)
    rewards = np.stack(rewards)
    return children, rewards


def generate_episodes(env, episodes, k, decay=1.0):
    """ Generate a random sequence of states starting from the solved state.

    @param env (Cube Object): A Cube object representing the environment.
    @param episodes (int): Number of episodes to be created.
    @param k (int): Length of backward moves.
    @returns states (Array[state]): Sequence of generated states. The shape of the array
                                    is (episodes * k_max, state.shape).
    @returns weights (Array): Array of weights. w[i] corresponds to the weight of states[i].
    @returns children (Array[state]): Sequence of states corresponding to the children of
                                      each of the generated states. The shape of the array
                                      is (episodes * k_max * num_acts, state.shape).
    @returns rewards (Array): Array of rewards. rewards[i] corresponds to the immediate
                              reward on transition to state children[i]
    """
    states, w = [], []

    # Create an environtment.
    cube = env()

    # Create `episodes` number of episodes.
    for _ in range(episodes):
        cube.reset()
        actions = np.random.randint(0, 12, k)
        states.extend((cube.step(act)[0] for act in actions))
        denom = 1 / (np.arange(1, k + 1) ** decay)
        w.extend(denom)

    # Expand each state to obtain children and rewards.
    children, rewards = expand_states(states, env)
    # # Discount rewards
    # gamma = np.ones(k, dtype=np.float32)
    # gamma[1:] = 0.99
    # gamma = np.multiply.accumulate(gamma)
    # gamma = np.tile(gamma, episodes)
    # rewards = np.array(rewards, dtype=np.float32)
    # rewards *= gamma.reshape(-1, 1)
    return jnp.array(states), jnp.array(w), jnp.array(children), jnp.array(rewards)


def make_targets(children, rewards, params):
    """ Generate target values.

    @param children (Array[state]): An array giving the children of the input states
                                    The shape of the array is (N * num_acts, state.shape).
    @param rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                            respective rewards.
    @param params (pytree): Model parameters for the prediction function.
    @returns vals (Array): An array giving the predicted values of each state.
    """
    # Run the states through the network in batches.
    batch_size = 1024
    vals = []
    for i in range(ceil(children.shape[0] / batch_size)):
        v = apply_fun(params, children[i * batch_size : (i + 1) * batch_size])
        vals.append(v)

    # Add rewards to make target values.
    vals = jnp.vstack(vals).reshape(rewards.shape) + rewards
    return jnp.max(vals, axis=1)


def batch_generator(rng, data, batch_size):
    """ Yields random batches of data.

    @param data (Dict):
    @param batch_size (int):
    @yields batch (Dict): Random batch of data of size `batch_size`.
    """
    choices = jnp.arange(data["X"].shape[0])
    while True:
        rng, sub_rng = jax.random.split(rng)
        idxs = jax.random.choice(sub_rng, choices, shape=(batch_size,), replace=False)
        yield (data["X"][idxs],
               data["y"][idxs],
               data["w"][idxs])


def beam_search(env, state, params, apply, max_depth=20):
    d = 0
    current = state.copy()
    while d < max_depth:
        d += 1
        children, _ = expand_states([current], env)
        Vs = apply(params, children).ravel()
        current = children[np.argmax(Vs)]
        if np.all(current == env.terminal_state):
            return True
    return False


#-------------------- optimizer and LR schedule --------------------#
step_size = 1e-4
decay_rate = 0.0
decay_steps = 1
step_fn = optimizers.inverse_time_decay(step_size=step_size,
                                        decay_rate=decay_rate,
                                        decay_steps=decay_steps)
opt_init, opt_update, get_params = optimizers.adam(step_size=step_fn)



#-------------------- params training utilities --------------------#
init_fun, apply_fun = model_fn()


@jax.jit
def l2_regularizer(params, reg=1e-5):
    """ Return the L2 regularization loss. """
    leaves, _ = tree_flatten(params)
    return reg * sum(jnp.vdot(x, x) for x in leaves)


@jax.jit
def hubber(x):
    delta = 2.0
    z = jnp.abs(x)
    return jnp.where(z < delta, 0.5 * (z ** 2), delta * (z - 0.5 * delta))


@jax.jit
def hubber_loss(params, batch):
    X, y, w = batch
    vals = apply_fun(params, X).ravel()
    return jnp.mean(hubber(vals - y) * w) + l2_regularizer(params, 1e-4)


@jax.jit
def update(i, opt_state, batch):
    """ Perform backpropagation and parameter update. """
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(hubber_loss)(params, batch)
    return loss, opt_update(i, grads, opt_state)


def train(rng, env, batch_size=128, num_epochs=5, num_iterations=21,
          num_samples=100, print_every=10, episodes=100, k_min=1, k_max=25,
          verbose=False, params_filepath=None, savepath=None):
    """
    Train the model function by generating simulations of random-play.
    On every epoch generate a new simulation and run multiple iterations.
    On every iteration evaluate the targets using the most recent model parameters
    and run multiple times through the dataset.
    At the end of every epoch check the performance and store the best performing params.
    If the performance drops then decay the step size parameter.

    @param rng (PRNGKey): A pseudo-random number generator.
    @param env (Cube Object): A Cube object representing the environment.
    @param batch_size (int): Size of minibatches used to compute loss and gradient during training.         [optional]
    @param num_epochs (int): The number of epochs to run for during training.                               [optional]
    @param num_iterations (int): The number of iterations through the generated episodes.                   [optional]
    @param num_samples (int): The number of times the dataset is reused.                                    [optional]
    @param print_every (int): An integer. Training progress will be printed every `print_every` iterations. [optional]
    @param episodes (int): Number of episodes to be created.                                                [optional]
    @param k_min (int): Minimum length of sequence of backward moves.                                       [optional]
    @param k_max (int): Maximum length of sequence of backward moves.                                       [optional]
    @param clip_norm (float): A scalar for gradient clipping.                                               [optional]
    @param verbose (bool): If set to false then no output will be printed during training.                  [optional]
    @param params_filepath (str): File path to save the model parameters.                                  [optional]
    @returns params (pytree): The best model parameters obatained during training.                          [optional]
    @returns loss_history (List): A list containing the mean loss computed during each iteration.           [optional]
    """
    # Initialize model parameters and optimizer state.
    rng, init_rng = jax.random.split(rng)
    input_shape = (-1,) + env.terminal_state.shape
    params = None
    if params_filepath is None:
        _, params = init_fun(init_rng, input_shape)
    else:
        params = list(jnp.load(params_filepath, allow_pickle=True))

    _solved_state = np.expand_dims(env.terminal_state, axis=0)
    _solved_state = jnp.array(_solved_state)
    # Set Numpy seed
    np.random.seed(17)

    # Generate test states
    test_set = []
    x = env()
    for _ in range(1000):
        x.reset()
        x.shuffle(np.random.randint(5, 15))
        test_set.append(x.state.copy())
    del x
    # Generate test episodes
    test_episodes_shape = (1000, 20)
    test_episodes = generate_episodes(env, *test_episodes_shape)[0]

    loss_history = []
    progress = []
    p_iteration_fmt = 'Iteration, {}, {}, {:.1f}, {:.3f}\n'
    p_epoch_fmt = 'Epoch {}, {}, {:.1f}, {:.3f}\n\n'

    # Begin training.
    decays = np.hstack([np.ones(20, dtype=np.float32),
                        np.linspace(1.0, 0.2, 64)])
    for e in range(num_epochs):
        decay = decays[e] if e < len(decays) else 0.2
        tic = time.time()
        opt_state = opt_init(params)

        # Generate data from random-play using the environment.
        states, w, children, rewards = generate_episodes(env, episodes, k_max, decay)

        # Train the model on the generated data. Periodically recompute the target values.
        epoch_mean_loss = 0.0
        for it in range(num_iterations):
            tic_it = time.time()

            # Make targets for the generated episodes using the most recent params and build a batch generator.
            params = get_params(opt_state)
            tgt_vals = make_targets(children, rewards, params)
            data = {"X" : states, "y" : tgt_vals, "w" : w}
            rng, sub_rng = jax.random.split(rng)
            train_batches = batch_generator(sub_rng, data, batch_size)

            # Run through the dataset and update model params.
            total_loss = 0.0
            for i in range(num_samples):
                batch = next(train_batches)
                loss, opt_state = update(it * num_samples + i, opt_state, batch)
                total_loss += loss

            # Book-keeping.
            iter_mean_loss = total_loss / num_samples
            epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)
            loss_history.append(iter_mean_loss)

            # Iteration verbose
            if it % print_every == 0 and verbose:
                toc_it = time.time()
                progress.append(p_iteration_fmt.format( it + 1,
                                                        num_iterations,
                                                        toc_it - tic_it,
                                                        iter_mean_loss))
                print(progress[-1], end='')

        # Record the time needed for a single epoch.
        toc = time.time()
        progress.append(
                p_epoch_fmt.format(
                    e + 1,
                    num_epochs,
                    toc - tic,
                    epoch_mean_loss
                ))
        # Do Value evaluation of TEST EPISIDES
        Vs = apply_fun(params, test_episodes).reshape(test_episodes_shape)
        Vmeans = np.mean(Vs, axis=0)
        Vsolved = float(apply_fun(params, _solved_state))
        print('Distance  0 states mean V:: {:.3f}'.format(Vsolved))
        for q, m in enumerate(Vmeans, 1):
            print('Distance {:2} states mean V: {:.3f}'.format(q, m))
        # Do Best First Search evaluation of TEST SET
        bs_solved = [beam_search(env, s, params, apply_fun) for s in test_set]
        bs_rate = sum(bs_solved) / len(bs_solved)
        print('GBFS solution rate: {:.2f}'.format(bs_rate))

        # Epoch verbose
        if verbose:
            print(progress[-1], end='')
        # Save parameters and output
        if savepath:
            pfilepath = os.path.join(savepath, 'params' + str(e))
            ofilepath = os.path.join(savepath, 'output' + str(e))
            jnp.save(pfilepath, params)
            with open(ofilepath, mode='w') as f:
                f.writelines(progress)

    return params, loss_history



if __name__ == "__main__":
    # from cube_model_naive import Cube as env
    from rubick import RubickCube as env
    rng = jax.random.PRNGKey(seed=17)

    ### Run training.
    params, loss_history = train(rng, env,
                                 batch_size=512,
                                 num_epochs=10,
                                 num_iterations=4,
                                 num_samples=48,
                                 print_every=1,
                                 episodes=1000,
                                 k_max=24,
                                 verbose=True,
                                 params_filepath=None)

    # epi = generate_episodes(rng, env, 1, 30)[0]
    # apply_fun(params, epi)

# from rubick import RubickCube as env
# rng = jax.random.PRNGKey(seed=17)

# states, w, children, rewards = generate_episodes(rng, env, 1, 4)