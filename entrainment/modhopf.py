import numpy as np
import matplotlib.pyplot as plt
from utils.floquet_utils import *
pi = np.pi

def _get_beta(x, C, degree):
    x = np.abs(x)
    X = np.concatenate([x ** p for p in range(degree, -1, -1 )], 0)
    return np.array([np.sum(C * X[:, i]) for i in range(X.shape[-1])], dtype = np.float32)

def _get_omega_choice(phi):
    return np.tanh(1e3 * (phi))

class ModHopf:
    def __init__(self, omega, degree = 15, Cpath = 'assets/out/plots/coef.npy'):
        self.omega = omega
        self.C = np.load(Cpath)
        self.degree = degree
        beta = _get_beta(self.omega, self.C, self.degree)
        self.mean = self.omega * np.abs(1 / (2 * self.beta * (1 - self.beta)))
        self.amplitude = self.omega * (1 - 2 * self.beta) / (2 * self.beta * (1 - self.beta))

    def hopf_simple_step(self, omega, mu, z):
        x, y = np.split(z, 2, 0)
        r = np.sqrt(x * x + y * y)
        dx = ((mu - r * r) * x - omega * y)
        dy = ((mu - r * r) * y + omega * x)
        z = np.concatenate([dx, dy], 0)
        return z

    def dif(self, X):
        x, y = np.split(X, 2, 0)
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        F = self.hopf_simple_step(omega, 1.0, X)
        return F

    def dif_per(self, X, q):
        x = X[0, 0]
        y = X[0, 0]
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q[0, 0]
        fy = (1.0 - r * r) * y + omega * x + q[1, 0]
        F = np.stack([fx, fy], 0)
        return F

    def dif_per1(self, X, q1):
        x = X[0, 0]
        y = X[0, 0]
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q1
        fy = (1.0 - r * r) * y + omega * x
        F = np.stack([fx, fy], 0)
        return F

    def dwdx(self, x, y):
        return -self.amplitude * 1e3 * y * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def dwdy(self, x, y):
        return self.amplitude * 1e3 * x * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def Jacobian(self, X):
        x = X[0, 0]
        y = X[1, 0]
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        f1x = 1 - 3 * x * x - y * y - y * self.dwdx(x, y)
        f1y = -2 * x * y - omega - y * self.dwdy(x, y)
        f2x = -2 * x * y + omega + x * self.dwdx(x, y)
        f2y = 1 - 3 * y * y - x * x + x * self.dwdy(x, y)
        #print(f1x.shape)
        #print(f1y.shape)
        #print(f2x.shape)
        #print(f2y.shape)

        J = np.concatenate([
            np.concatenate([f1x, f1y], 0).T,
            np.concatenate([f2x, f2y], 0).T
        ], 0)
        #print(J.shape)
        return J

class ModHopf_rescale:
    def __init__(self, omega, timescale, degree = 15, Cpath = '../assets/out/plots/coef.npy'):
        self.omega = omega
        #print(self.omega)
        self.C = np.load(Cpath)
        self.degree = degree
        self.beta = _get_beta(self.omega, self.C, self.degree)
        #print(self.beta)
        self.mean = self.omega * np.abs(1 / (2 * self.beta * (1 - self.beta)))
        self.amplitude = self.omega * (1 - 2 * self.beta) / (2 * self.beta * (1 - self.beta))
        #print(self.mean)
        #print(self.amplitude)
        self.timescale = timescale

    def hopf_simple_step(self, omega, mu, z):
        x, y = np.split(z, 2, 0)
        r = np.sqrt(x * x + y * y)
        dx = ((mu - r * r) * x - omega * y)
        dy = ((mu - r * r) * y + omega * x)
        z = np.concatenate([dx, dy], 0)
        return z

    def dif(self, X):
        x, y = np.split(X, 2, 0)
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        F = self.hopf_simple_step(omega, 1.0, X)
        return F * self.timescale

    def dif_per(self, X, q):
        x, y = np.split(X, 2, 0)
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        F = self.hopf_simple_step(omega, 1.0, X)
        fx, fy = np.split(F, 2, 0)
        fx = fx + q[0]
        fy = fy + q[1]
        F = np.concatenate([fx, fy], 0)
        return F * self.timescale

    def dif_per1(self, X, q1):
        x = X[0, 0]
        y = X[0, 0]
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        r = np.sqrt(x * x + y * y)
        fx = (1.0 - r * r) * x - omega * y + q1
        fy = (1.0 - r * r) * y + omega * x
        F = np.stack([fx, fy], 0)
        return F * self.timescale

    def dwdx(self, x, y):
        return -self.amplitude * 1e3 * y * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def dwdy(self, x, y):
        return self.amplitude * 1e3 * x * (
            1 - np.tanh(1e3 * np.arctan2(y, x)) ** 2
        ) / ((x ** 2 + y ** 2) * 2)

    def Jacobian(self, X):
        x = X[0, 0]
        y = X[1, 0]
        phi = np.arctan2(y, x)
        omega = (self.mean + self.amplitude * _get_omega_choice(phi)) / 2
        f1x = 1 - 3 * x * x - y * y - y * self.dwdx(x, y)
        f1y = -2 * x * y - omega - y * self.dwdy(x, y)
        f2x = -2 * x * y + omega + x * self.dwdx(x, y)
        f2y = 1 - 3 * y * y - x * x + x * self.dwdy(x, y)
        #print(f1x.shape)
        #print(f1y.shape)
        #print(f2x.shape)
        #print(f2y.shape)

        J = np.concatenate([
            np.concatenate([f1x, f1y], 0).T,
            np.concatenate([f2x, f2y], 0).T
        ], 0) * self.timescale
        #print(J.shape)
        return J
