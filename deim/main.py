# Copyright (C) 2020 by Triet Pham
#
# Apply DEIM to approximate a nonlinear function:
#   g(x;mu) = (1 - x) * cos(3*pi*mu*(x + 1)) * e^(-(1 + x)*mu)

import numpy as np
import matplotlib.pyplot as plt
from DEIM import DEIMApproximation


class TestCase(DEIMApproximation):
    def exact_function(self, x, mu):
        return (1 - x)*np.cos(3*np.pi*mu*(x+1))*np.exp(-(1+x)*mu)

# Initialization
x = np.linspace(-1, 1, num=100)  # \mathcal{N}
mu = np.linspace(1, np.pi, num=51)  # M

# Use more basis functions in TestCase(x, mu, basis=8) 
# to increase the accuracy of DEIM approximation
test = TestCase(x, mu, basis=8)
test.offline()
test.plot_singular_values(fig_num=1)

# Visualization
test.online(mu=1.17)
test.plot_exact_function(fig_num=2)
test.plot_approx_function(fig_num=2)

plt.show()
