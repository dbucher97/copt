import jax
import numpy as np

@jax.jit
def fwht(x):
    d = x.shape[0]
    h = 2
    while h <= d:
        x = x.reshape((h, -1))
        a = x[::2]
        b = x[1::2]
        x = x.at[::2].set((a + b))
        x = x.at[1::2].set((a - b))
        h *= 2

    return x.flatten() / np.sqrt(d)

if __name__ == "__main__":
    import jax.numpy as jnp
    import time
    x = jnp.zeros(2 ** 20)
    x = x.at[0].set(1.)
    y = fwht(x)

    a = time.time()
    for i in range(100):
        fwht(x).block_until_ready()
    b = time.time()
    print((b - a) / 100)
    print(y)

