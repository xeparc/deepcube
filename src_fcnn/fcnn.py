"""Fully connected neural networks"""
import numpy as np
import jax.numpy as jnp
import jax

from jax.experimental.stax import Conv, Relu, MaxPool, Flatten, Dense, serial
from jax.nn import log_softmax

def fc_net():
    vhead_out_dim, phead_out_dim = 1, 12
    serial_init, serial_apply = serial(Flatten,
                                       Dense(4096), Relu,
                                       Dense(2048), Relu)
    vhead_init, vhead_apply = serial(Dense(512), Relu, Dense(vhead_out_dim))
    phead_init, phead_apply = serial(Dense(512), Relu, Dense(phead_out_dim))

    def init_fun(rng, input_shape):
        rng, vhead_rng, phead_rng = jax.random.split(rng, 3)
        serial_out_shape, serial_params = serial_init(rng, (-1,) + input_shape)
        vhead_out_shape, vhead_params = vhead_init(vhead_rng, serial_out_shape)
        phead_out_shape, phead_params = phead_init(phead_rng, serial_out_shape)
        out_shape = (vhead_out_shape, phead_out_shape)
        params = [serial_params, vhead_params, phead_params]
        return out_shape, params

    def apply_fun(params, inputs):
        serial_params, vhead_params, phead_params = params
        serial_out = serial_apply(serial_params, inputs)
        vhead_out = vhead_apply(vhead_params, serial_out)
        phead_out = log_softmax(phead_apply(phead_params, serial_out))
        out = (vhead_out, phead_out)
        return out
    return init_fun, apply_fun


def value_policy_approximator():
    """ Return (value, policy) tuple for an input state. """
    serial_init, serial_apply = serial(Flatten,
                                       Dense(2048), Relu,
                                       Dense(1024), Relu)
    vhead_init, vhead_apply = serial(Dense(512), Relu, Dense(1))
    phead_init, phead_apply = serial(Dense(512), Relu, Dense(12))

    def init_fun(rng, input_shape):
        rng, vhead_rng, phead_rng = jax.random.split(rng, 3)
        serial_out_shape, serial_params = serial_init(rng, input_shape)
        vhead_out_shape, vhead_params = vhead_init(vhead_rng, serial_out_shape)
        phead_out_shape, phead_params = phead_init(phead_rng, serial_out_shape)
        out_shape = (vhead_out_shape, phead_out_shape)
        params = [serial_params, vhead_params, phead_params]
        return out_shape, params

    def apply_fun(params, inputs):
        serial_params, vhead_params, phead_params = params
        serial_out = serial_apply(serial_params, inputs)
        vhead_out = vhead_apply(vhead_params, serial_out)
        phead_out = log_softmax(phead_apply(phead_params, serial_out))
        return (vhead_out, phead_out)

    return init_fun, apply_fun


def value_appoximator():
    """
    Approximator for value of the input state.

    Returns
    -------
    (init, apply) tuple
    """
    init, apply = serial(Flatten,
                         Dense(2048), # 1024
                         Relu,
                         Dense(1024), # 512
                         Relu,
                         Dense(512),  # 256
                         Relu,
                         Dense(1))
    return init, apply


init_fun, apply_fun = fc_net()
