import numpy as np
import matplotlib.pyplot as plt
from DEIM import DEIMApproximation


class TestCase(DEIMApproximation):
    def exact_function(self, x, mu):
        return (1 - x)*np.cos(3*np.pi*mu*(x+1))*np.exp(-(1+x)*mu)

# Initialization
x = np.linspace(-1, 1, num=100)  # \mathcal{N}
mu = np.linspace(1, np.pi, num=51)  # M

# test = DEIMApproximation(x, mu, basis=15)
test = TestCase(x, mu, basis=15)
test.offline()
test.plot_singular_values(fig_num=1)

# Visualization
test.online(1.17)
test.plot_exact_function(2)
test.plot_approx_function(2)

plt.show()
