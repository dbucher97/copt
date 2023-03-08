import numpy as np
import matplotlib.pyplot as plt

def get_prob(n):
    x = np.arange(2 ** (n - 1))

    @np.vectorize
    def prob(t):
        d = np.pi * (x - 2 ** n * t)
        s1 = np.sin(d)
        s2 = np.sin(d * 2 ** (-n))
        ic = np.isclose(d, 0)
        v = np.abs(np.divide(s1, s2, where=np.logical_not(ic)) ) ** 2
        return (np.sum(v)) / 2 ** (2 * n) + np.sum(ic)

    return prob


# y = np.linspace(0, 1, 1000)
#
# for n in range(8, 1, -2):
#     plt.plot(y, get_prob(n)(y))
# plt.show()

N = 6
x = np.linspace(-1, 1, 2 ** N)
y = np.linspace(-1, 1, 2 ** N)

X, Y = np.meshgrid(x, y)

print(x)


plt.matshow(get_prob(6)(((X**2 + Y ** 2) + 2) / 4 ))
plt.show()
