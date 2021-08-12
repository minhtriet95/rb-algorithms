# Copyright (C) 2020 by Triet Pham
#
# Reference:
#   S. Chaturantabut and D.C. Sorensen. Nonlinear model reduction
#   via discrete empirical interpolation.
#   SIAM Journal on Scientific Computing, 32(5):2737-2764, 2010.
#
# Base class of DEIM approximation.

import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
from decorators.timer import show_execution_time


class DEIMApproximation:
    # Constructor for DEIM approximation class
    def __init__(self, x_list, mu_list, basis):
        # self.use_latex(True)
        self.x_list = x_list
        self.mu_list = mu_list
        self.basis = basis

        # Create a matrix of snapshots
        f_array = self.array_expression()

        # Perform SVD, store singular vectors and values
        U, Sigma, Vh = np.linalg.svd(f_array)
        self.U = U
        self.Sigma = Sigma

    def exact_function(self, x, mu):
        """Please implement this method to return the exact (original) function.

        Arguments:
            x  -- list of spatial points
            mu -- list of parameters
        """
        raise NotImplementedError("Please implement this method in your class")

    def array_expression(self):
        """Return an array representation (snapshots) of the given function."""
        s = list()
        for mu in self.mu_list:
            s.append(self.exact_function(self.x_list, mu))
        return np.asarray(s).transpose()

    def update_measurement_matrix(self, iter, index):
        """Construct and update the measurement matrix P.

        Return: 
            first iteration -- convert P from a list to a numpy.ndarray
            other iteration -- P is already a numpy.ndarray
        """
        P = self.P
        if iter == 0:
            # P is a normal array (list)
            e_vector = np.zeros(len(self.x_list))
            e_vector[index] = 1
            P.append(e_vector)
            return np.asarray(P).transpose()
        else:
            # P is a numpy.ndarray
            e_vector = np.zeros((len(self.x_list), 1))
            e_vector[index] = 1
            P = np.append(P, e_vector, axis=1)
            return P

    def calculate_coefficients(self, iter):
        """Calculate the coefficient vector c.

        Return:
            c = (P^T * U)^{-1} * P^T * next_basis_vector
        """
        T = np.matmul(self.P.transpose(), self.U[:, :iter+1])
        c = np.matmul(np.linalg.inv(T), self.P.transpose())
        c = np.matmul(c, self.U[:, iter+1])
        return np.asarray(c)

    @show_execution_time
    def offline(self):
        """Find the residual and update P matrix.

        Procedure:
            First iteration (i = 1)
                1. Find 1st index = argmax(first_basis_vector)
                2. Initialize the measurement matrix P
                3. Calculate coefficient c
                4. Compute the residual for next iteration (i = 2)
            Other iteration (i >= 2)
                1. Find indices = argmax(previous_residual)
                2. Recompute the measurement matrix
                3. Calculate coefficient vector c
                4. Compute the residual for next iteration
        """
        for q in range(self.basis):
            if q == 0:
                # Construct the measurement matrix with the first index i = 1
                index_1 = np.abs(self.U[0, :]).argmax()
                self.P = list()
                self.P = self.update_measurement_matrix(0, index_1)
                c = self.calculate_coefficients(q)
                # Find the residual for next iteration
                residual = self.U[:, q+1] - np.dot(self.U[:, :q+1], c)
            else:
                # Find other indices (i = 2,...,Q_basis)
                index_p = np.abs(residual).argmax()
                self.P = self.update_measurement_matrix(q, index_p)
                c = self.calculate_coefficients(q)
                # Update the residual for next iteration
                residual = self.U[:, q+1] - np.dot(self.U[:, :q+1], c)

    @show_execution_time
    def online(self, mu):
        """Find the online coefficients corresponding to mu and approximate f.

        Procedure:
            1. Find c(\mu) = (P^T * U)^{-1} * P^T * f(\mu)
            2. Approximate the original function by projection
        Note:
            c (offline) != c_mu (online)
        """
        self.mu = mu
        # Step 1: c(\mu) = (P^T * U)^{-1} * P^T * f(\mu)
        f = self.exact_function(self.x_list, mu)
        f = np.asarray(f).transpose()
        T = np.matmul(self.P.transpose(), self.U[:, :self.basis])
        c_mu = np.matmul(np.linalg.inv(T), self.P.transpose())
        c_mu = np.dot(c_mu, f)
        # Step 2: \hat(f(\mu)) = U * c(\mu)
        approximation = np.dot(self.U[:, :self.basis], c_mu)
        self.approximate_function = approximation

    def use_latex(self, activate):
        """Activate LaTeX engine to use its fonts and syntax."""
        if activate == True:
            from matplotlib import rc
            rc('font', **{'family': 'serif', 'sans-serif': ['Helvetica']})
            rc('text', usetex=True)
        else:
            pass

    def plot_exact_function(self, fig_num):
        """Plot the exact (original) function."""
        plt.figure(fig_num)
        original = self.exact_function(self.x_list, self.mu)
        plt.plot(self.x_list, original, 'b-')
        plt.title(f"$\mu = {self.mu}$, basis = ${self.basis}$", fontsize=16)
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$s(x;\mu)$", fontsize=16)
        plt.grid()

    def plot_approx_function(self, fig_num):
        """Plot the approximate function by DEIM.

        Note: 
            Must call in the same figure with plot_exact_function()
        """
        plt.figure(fig_num)
        plt.plot(self.x_list, self.approximate_function, 'r--')
        plt.legend(['exact function', 'DEIM approximation'], fontsize=14)

    def plot_singular_values(self, fig_num):
        """Plot the singular values (sigmas)."""
        plt.figure(fig_num)
        Sigma = self.Sigma
        interval = len(self.mu_list)
        plt.plot(np.linspace(1, interval, interval), Sigma, 'o-')
        plt.yscale("log")
        # plt.xlabel("Snapshots", fontsize=16, usetex=True)
        # plt.ylabel("Singular values", fontsize=16, usetex=True)
        plt.xlabel("Snapshots", fontsize=16)
        plt.ylabel("Singular values", fontsize=16)
        plt.grid()


if __name__ == "__main__":

    # create a TestCase class to implement exact_function() method
    class TestCase(DEIMApproximation):
        def exact_function(self, x, mu):
            return (1 - x)*np.cos(3*np.pi*mu*(x+1))*np.exp(-(1+x)*mu)

    # Initialization
    x = np.linspace(-1, 1, num=100)  # \mathcal{N}
    mu = np.linspace(1, np.pi, num=51)  # M

    # test = DEIMApproximation(x, mu, basis=10)
    test = TestCase(x, mu, basis=15)
    test.offline()
    test.plot_singular_values(fig_num=1)

    # Visualization
    # mu_test = [1.17, 1.5, 2.3, 3.1]
    mu_test = [1.17]
    for figure, mu in enumerate(mu_test):
        figure += 2
        test.online(mu)
        test.plot_exact_function(figure)
        test.plot_approx_function(figure)

    plt.show()
