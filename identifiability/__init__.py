import os
import multiprocessing
import jax

cpu_count = multiprocessing.cpu_count()

print("Setting JAX to CPU and 64-bit precision.")
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

print(f"Setting XLA to use {cpu_count} CPU threads.")
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count}'
