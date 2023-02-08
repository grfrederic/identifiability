from time import time

import jax
import jax.numpy as jnp
import blackjax


def make_safe_log_prob_fn(
    prior_log_prob,
    cond_log_prob,
):
    def log_prob_fn(parameters):
        prior = prior_log_prob(parameters)

        return jax.lax.cond(
            jnp.isfinite(prior),
            lambda: prior + cond_log_prob(parameters),
            lambda: prior,
        )
    return log_prob_fn


def normal_error_log_prob(xs, ys, sigma=1.0):
    return -0.5 * jnp.square((xs - ys) / sigma).sum()


def run_mcmc(
    parameters_init,
    log_prob_fn,
    sigma,
    num_chains, num_steps, num_burn_in, thinning,
    seed=0,
):
    rmh = blackjax.rmh(log_prob_fn, sigma)
    initial_state = rmh.init(parameters_init)
    rmh_kernel = jax.jit(rmh.step)

    keys = jax.random.split(jax.random.PRNGKey(seed), num_chains)

    t0 = time()
    states = inference_loop_multiple_chains(
        keys,
        rmh_kernel,
        initial_state,
        num_steps,
    )
    _ = states.position.block_until_ready()
    t1 = time()
        
    samples_by_chain = states.position
    log_probs_by_chain = states.log_probability
    print(f"Done, took: {t1 - t0:.2f}s")
    print(f"{num_steps / (t1 - t0):.2f} samples / sec / chain")

    samples_by_chain = samples_by_chain[:, num_burn_in::thinning]
    log_probs_by_chain = log_probs_by_chain[:, num_burn_in::thinning]

    return samples_by_chain, log_probs_by_chain


def mcmc_report(samples_by_chain, parameters_names=None):
    # r-hats and effective samples
    efss = blackjax.diagnostics.effective_sample_size(samples_by_chain)
    rs = blackjax.diagnostics.potential_scale_reduction(samples_by_chain)

    for i, (r, efs) in enumerate(zip(rs, efss)):
        name = f'rate {i}' if parameters_names is None else parameters_names[i]
        print(f"{name} r-hat: {float(r):.3f} (eff. {efs:4.1f})")


def blackjax_inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

inference_loop_multiple_chains = jax.pmap(
    blackjax_inference_loop,
    in_axes=(0, None, None, None),
    static_broadcasted_argnums=(1, 3),
)
