import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(1234)
noise_level = 0.1

# def f(x, noise_level=noise_level):
#     return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
#             + np.cos(7 * x[0]) * np.log(1 + x[0] ** 2)\
#             + np.random.randn() * noise_level

def f(x, noise_level=0.1):
    if x[0] < 5:
        return np.sin(x[0]) + 0.5 * x[0]
    elif 5 <= x[0] < 80:
        return np.sin(x[0]) + 0.5 * x[0] + np.random.randn() * noise_level * 1000
    else:
        return np.sin(x[0]) - 0.5 * x[0] + np.random.randn() * noise_level * 10

def branin(x, noise_level=noise_level):
    x1, x2 = x
    term1 = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + (5 / np.pi) * x1 - 6) ** 2
    term2 = 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1)
    result = term1 + term2 + 10
    if noise_level > 0:
        result += np.random.randn() * noise_level
    return result

