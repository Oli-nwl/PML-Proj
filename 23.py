import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.contrib.gp import models, kernels
import matplotlib.pyplot as plt
import arviz as az

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
pyro.set_rng_seed(42)

def g(x):
    return -(torch.sin(6 * np.pi * x))**2 + 6 * x**2 - 5 * x**4 + 1.5

# Create dataset
l = 30
x = torch.linspace(0, 1, l).to(device)
y = g(x) + torch.randn(l, device=device) * 0.1  # Add noise with std=0.1

# Split into training and evaluation sets
perm = torch.randperm(l, device=device)
train_idx, test_idx = perm[:20], perm[20:]
X_train, y_train = x[train_idx], y[train_idx]
X_test, y_test = x[test_idx], y[test_idx]

class GPModel(pyro.nn.PyroModule):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

        # Define the RBF kernel
        self.kernel = kernels.RBF(input_dim=1)

        # Define the GP model
        self.gp = models.GPRegression(X, y, self.kernel, noise=torch.tensor(0.01, device=device))

    def model(self):
        # Sample from priors
        lengthscale = pyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        variance = pyro.sample("variance", dist.LogNormal(0.0, 1.0))

        # Update the kernel parameters
        self.kernel.lengthscale = lengthscale
        self.kernel.variance = variance

        # Return the GP model
        return self.gp.model()

# Initialize the model
gp_model = GPModel(X_train, y_train)

# Run NUTS sampler
nuts_kernel = pyro.infer.NUTS(gp_model.model)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run()

# Detach and convert MCMC samples to numpy arrays
detached_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

# Convert detached samples to Arviz InferenceData
posterior = az.from_dict(posterior=detached_samples)

# Compute and print diagnostics
summary = az.summary(posterior, var_names=["lengthscale", "variance"])
print(summary)

# Plot trace and posterior distribution
az.plot_trace(posterior, var_names=["lengthscale", "variance"])
plt.savefig("trace_plot.png")
plt.close()

# Check R-hat statistic
rhat = az.rhat(posterior, var_names=["lengthscale", "variance"])
print(f"R-hat (lengthscale): {rhat['lengthscale'].item():.4f}")
print(f"R-hat (variance): {rhat['variance'].item():.4f}")

# Effective sample size
ess = az.ess(posterior, var_names=["lengthscale", "variance"])
print(f"Effective sample size (lengthscale): {ess['lengthscale'].item():.4f}")
print(f"Effective sample size (variance): {ess['variance'].item():.4f}")

# Make predictions using the posterior samples
X_plot = torch.linspace(0, 1, 100, device=device)
predictions = []

for lengthscale, variance in zip(mcmc.get_samples()["lengthscale"], mcmc.get_samples()["variance"]):
    gp_model.kernel.lengthscale = lengthscale
    gp_model.kernel.variance = variance
    with torch.no_grad():
        mean, _ = gp_model.gp(X_plot)
    predictions.append(mean.cpu().numpy())

predictions = np.stack(predictions)
mean_prediction = predictions.mean(axis=0)
std_prediction = predictions.std(axis=0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_plot.cpu().numpy(), g(X_plot).cpu().numpy(), 'k', label='True function')
plt.plot(X_plot.cpu().numpy(), mean_prediction, 'b', label='GP mean')
plt.fill_between(X_plot.cpu().numpy(), 
                 mean_prediction - 2 * std_prediction, 
                 mean_prediction + 2 * std_prediction, 
                 alpha=0.2, color='b', label='95% CI')
plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), c='r', marker='o', label='Training data')
plt.scatter(X_test.cpu().numpy(), y_test.cpu().numpy(), c='g', marker='s', label='Test data')
plt.legend()
plt.title('GP Regression (RBF Kernel) - NUTS')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("gp_regression_plot.png")
plt.close()

print("Sampling and diagnostics completed. Check the generated plot files for visualizations.")
