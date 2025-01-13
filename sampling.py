import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from math import sin, exp

# Utility function for timeout (placeholder)
def timeout(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@timeout
def problem1_inversion(n_samples=1):
    # Distribution function for rejection sampling
    def target_density(x):
        return np.exp(x**2 - 1) / (np.exp(-1)) if 0 < x < 1 else 0

    # Proposal distribution: uniform on [0, 1]
    proposal_density = lambda x: 1 if 0 <= x <= 1 else 0

    # Proposal constant: find a good constant such that c * proposal dominates target
    c = np.exp(1)

    samples = []
    while len(samples) < n_samples:
        x = uniform.rvs(loc=0, scale=1)  # Sample from uniform [0, 1]
        u = uniform.rvs(loc=0, scale=1)  # Sample uniform for acceptance
        if u <= target_density(x) / (c * proposal_density(x)):
            samples.append(x)

    return np.array(samples)

# Part 2: Generate 100000 samples
problem1_samples = problem1_inversion(n_samples=100000)

# Plot histogram
plt.hist(problem1_samples, bins=100, density=True, alpha=0.5, label="Samples")

# True density
x_vals = np.linspace(0, 1, 1000)
y_vals = [np.exp(x**2 - 1) / (np.exp(-1)) for x in x_vals]
plt.plot(x_vals, y_vals, label="True density")

plt.legend()
plt.show()

# Part 3: Compute the integral
integral_samples = (sin(problem1_samples) * 2 * np.exp(problem1_samples**2)) / (
    problem1_samples * (np.exp(-1))
)
problem1_integral = np.mean(integral_samples)

# Part 4: Compute 95% confidence interval
sample_variance = np.var(integral_samples)
n = len(integral_samples)
confidence_bound = 1.96 * np.sqrt(sample_variance / n)
problem1_interval = (
    problem1_integral - confidence_bound,
    problem1_integral + confidence_bound,
)

# Part 5: Additional distribution
@timeout
def problem1_inversion_2(n_samples=1):
    # Target density for part 5
    def target_density(x):
        return 20 * x * np.exp(20 - 1 / x) / 20 if 0 < x < 1 else 0

    # Proposal distribution: uniform [0, 1]
    proposal_density = lambda x: 1 if 0 <= x <= 1 else 0

    # Proposal constant
    c = 20 * np.exp(19)

    samples = []
    while len(samples) < n_samples:
        x = uniform.rvs(loc=0, scale=1)
        u = uniform.rvs(loc=0, scale=1)
        if u <= target_density(x) / (c * proposal_density(x)):
            samples.append(x)

    return np.array(samples)

# Example usage:
problem1_samples_2 = problem1_inversion_2(n_samples=100000)
