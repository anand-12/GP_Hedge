from test_functions import f, branin
from scipy.optimize import minimize

result = minimize(f, x0=[5], bounds=[(0, 30)])

print(f"Minimum value: {result.fun} at x = {result.x}")
