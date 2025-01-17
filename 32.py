import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Generate data points
def g(x):
    return -(np.sin(6 * np.pi * x))**2 + 6 * x**2 - 5 * x**4 + 1.5

l = 30
x = np.linspace(0, 1, l)
y = g(x)

# Randomly partition into training and evaluation sets
indices = np.random.permutation(l)
train_indices = indices[:20]
eval_indices = indices[20:]

x_train = x[train_indices]
y_train = y[train_indices]
x_eval = x[eval_indices]
y_eval = y[eval_indices]

# Define the model
with pm.Model() as gp_model:
    # Priors for the hyperparameters
    length_scale = pm.LogNormal("length_scale", mu=0, sigma=1)
    signal_variance = pm.LogNormal("signal_variance", mu=0, sigma=1)
    noise_variance = pm.HalfNormal("noise_variance", sigma=0.1)

    # Specify the covariance function
    cov_func = signal_variance * pm.gp.cov.ExpQuad(1, ls=length_scale)

    # Specify the GP
    gp = pm.gp.Marginal(cov_func=cov_func)

    # Specify the GP likelihood
    _ = gp.marginal_likelihood("y", X=x_train[:, None], y=y_train, noise=noise_variance)

# Sampling
def run_sampling(model, n_draws, n_tune):
    with model:
        trace = pm.sample(draws=n_draws, tune=n_tune, chains=1, return_inferencedata=True)
    return trace

# Initial sampling with default values
n_draws_initial = 1000
n_tune_initial = 1000
trace_initial = run_sampling(gp_model, n_draws_initial, n_tune_initial)

# Diagnostics
def print_diagnostics(trace):
    summary = az.summary(trace, round_to=2)
    print("Summary:")
    print(summary)
    
    print("\nR-hat values:")
    print(az.rhat(trace))
    
    print("\nEffective Sample Size:")
    print(az.ess(trace))

print("Initial Diagnostics:")
print_diagnostics(trace_initial)

# Plot traces
az.plot_trace(trace_initial)
plt.tight_layout()
plt.savefig('trace_plot_initial.png')
plt.close()

# Based on the initial diagnostics, adjust the number of samples if needed
n_draws_final = 2000  # Increase if needed
n_tune_final = 1000   # Increase if needed

trace_final = run_sampling(gp_model, n_draws_final, n_tune_final)

print("\nFinal Diagnostics:")
print_diagnostics(trace_final)

# Plot final traces
az.plot_trace(trace_final)
plt.tight_layout()
plt.savefig('trace_plot_final.png')
plt.close()

# Plot posterior distributions
az.plot_posterior(trace_final)
plt.tight_layout()
plt.savefig('posterior_plot.png')
plt.close()

# Extract posterior samples
posterior_samples = az.extract(trace_final)

# Function to compute GP mean and variance
def gp_predict(x_new, x_train, y_train, length_scale, signal_variance, noise_variance):
    K = signal_variance * np.exp(-0.5 * ((x_train[:, None] - x_train[None, :]) / length_scale) ** 2)
    K += noise_variance * np.eye(len(x_train))
    K_star = signal_variance * np.exp(-0.5 * ((x_new[:, None] - x_train[None, :]) / length_scale) ** 2)
    K_star_star = signal_variance * np.exp(-0.5 * ((x_new[:, None] - x_new[None, :]) / length_scale) ** 2)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    v = np.linalg.solve(L, K_star.T)

    mean = K_star @ alpha
    var = K_star_star - v.T @ v

    return mean, np.diag(var)

# Predict using posterior samples
x_plot = np.linspace(0, 1, 100)
y_means = []
y_vars = []

for i in range(100):  # Use 100 posterior samples
    length_scale = posterior_samples.length_scale[i]
    signal_variance = posterior_samples.signal_variance[i]
    noise_variance = posterior_samples.noise_variance[i]
    
    mean, var = gp_predict(x_plot[:, None], x_train[:, None], y_train, length_scale, signal_variance, noise_variance)
    y_means.append(mean)
    y_vars.append(var)

y_mean = np.mean(y_means, axis=0)
y_std = np.sqrt(np.mean(y_vars, axis=0) + np.var(y_means, axis=0))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_plot, g(x_plot), 'r', label='True function')
plt.plot(x_plot, y_mean, 'b', label='GP mean')
plt.fill_between(x_plot, y_mean - 1.96 * y_std, y_mean + 1.96 * y_std, alpha=0.2)
plt.scatter(x_train, y_train, c='k', label='Training data')
plt.scatter(x_eval, y_eval, c='g', label='Evaluation data')
plt.legend()
plt.title('Gaussian Process Regression with NUTS Sampling')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("gp_nuts_result.png")
plt.close()

print("Sampling and diagnostics completed. Check the generated plot files for visualizations.")