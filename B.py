import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
import torch
import matplotlib.pyplot as plt

# Step 1: Generate the Dataset
np.random.seed(42)
torch.manual_seed(42)

l = 30  # Total number of points
xi = np.linspace(0, 1, l)
def g(x):
    return -(np.sin(6 * np.pi * x))**2 + 6 * x**2 - 5 * x**4 + 1.5

epsilon = np.random.normal(0, 0.01, l)
yi = g(xi) + epsilon

# Partition into training and testing
train_indices = np.random.choice(l, size=20, replace=False)
test_indices = np.setdiff1d(np.arange(l), train_indices)
xi_train, yi_train = xi[train_indices], yi[train_indices]
xi_test, yi_test = xi[test_indices], yi[test_indices]

# Convert data to torch tensors
xi_train_torch = torch.tensor(xi_train, dtype=torch.float32)
yi_train_torch = torch.tensor(yi_train, dtype=torch.float32)
xi_test_torch = torch.tensor(xi_test, dtype=torch.float32)

# Step 2: Define the GP Model
kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(1.0), lengthscale=torch.tensor(1.0))
gpr = gp.models.GPRegression(xi_train_torch.unsqueeze(-1), yi_train_torch, kernel, noise=torch.tensor(0.1))

# Step 3: Optimize Hyperparameters using MAP
optimizer = pyro.optim.Adam({"lr": 0.01})
loss_fn = Trace_ELBO()
svi = SVI(gpr.model, gpr.guide, optimizer, loss=loss_fn)

# Train the model
num_steps = 2000
for step in range(num_steps):
    loss = svi.step()
    if step % 500 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")

# Predict on test data
mean, cov = gpr(xi_test_torch.unsqueeze(-1), full_cov=True)
mean = mean.detach().numpy()
std = torch.sqrt(torch.diag(cov)).detach().numpy()

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(xi, g(xi), label="True Function (g(x))", linestyle="--", color="blue")
plt.scatter(xi_train, yi_train, label="Training Points", color="red", alpha=0.7)
plt.scatter(xi_test, yi_test, label="Test Points", color="green", alpha=0.7)
plt.plot(xi_test, mean, label="GP Predictions (MAP)", color="black")
plt.fill_between(xi_test, mean - 2 * std, mean + 2 * std, color="gray", alpha=0.3, label="Confidence Interval")
plt.legend()
plt.title("Gaussian Process (MAP Estimation)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid()
plt.show()

# Step 4: Replace MAP with NUTS
nuts_kernel = NUTS(gpr.model)
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
mcmc.run()
posterior_samples = mcmc.get_samples()

# Explain results in detail
