import jax.numpy as jnp


def const_0(t):
    return jnp.zeros_like(t)


def const_1(t):
    return jnp.ones_like(t)


# on-off, for training
def make_pulse(h):
    "signal active till `h`"
    return lambda t: t < h


def make_pulse_delayed(h):
    "signal active from `1` to `h`"
    return lambda t: (t >= 1) * (t < h + 1)


# complex, for testing
def test_blocky(t):
    return (t >= 1) * (t < 2) + (t >= 4) * (t < 6) + (t >= 7)


def test_wiggly(t):
    return 2.7 * jnp.sin(t / 2) ** 4 * jnp.exp(-t / 10)
