# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:14:05 2024

@author: lisy
"""


pip install tensorflow
pip install gpflow


import gpflow
import numpy as np
import matplotlib.pylab as plt

import heteroskedastic  # Definition of model

## Create synthetic dataset
## ------------------------
N = 50  # size of training data
np.random.seed(1324)  # Ensure same random numbers every time

# Generate training and testing datapoints
xmax = 20
X = np.random.rand(N,1) * xmax
Xtest = np.linspace(0, xmax*2, 500).reshape(-1, 1)
Z = X.copy()

# Generate labels
y = 0.8 / (1 + np.exp(-1 * X.flatten()))
y = np.random.rand(y.size) > y
y = y.astype(int)
# Ensure existence of two classes:
# If all labels are 1, set the first one as 0, vice versa
if np.argmax(y) == 0:
    y[0] = 1
elif np.argmin(y) == 1:
    y[0] = 0

# One-hot vector encoding
# The first column represents the first class: y = 0 -- Y = 1
# The second column represents the second class: y = 1 -- Y = 1
Y = np.zeros((y.size, 2))
Y[:,0], Y[:,1] = 1-y, y



## Setup heteroskedastic regression
## ================================
a_eps = 0.1  # Hyperparameter from Gamma distribution (for Dirichlet prior)
alpha = Y + a_eps

# Transform original labels to log-normal space
s2_tilte = np.log(1/alpha + 1)
Y_tilte = np.log(alpha) - s2_tilte / 2

######unclear
ymean = np.log(Y.mean(0)) + np.mean(Y_tilte-np.log(Y.mean(0)))
Y_tilte = Y_tilte - ymean




## Setup GP and optimise hyperparameters
## Hyper: alpha_eps (for Dirichlet), a (marginal variance for RBF), l (length-scale for RBF)
## =========================================================================================
kernel = gpflow.kernels.RBF(1)
model = heteroskedastic.SGPRh(X, Y_tilte, kern=kernel, Z=Z, sn2=s2_tilte)
model.kern.lengthscale = np.std(X)
model.kern.variance = np.var(Y_tilte)
# Set parameters for kernel

# Optimise the model
opt = gpflow.train.ScipyOptimizer()
print('\nloglik (before) =', model.compute_log_likelihood())
print('ampl =', model.kern.variance.read_value())
print('leng =', model.kern.lengthscales.read_value())
opt.minimize(model)
print('loglik  (after) =', model.compute_log_likelihood())
print('ampl =', model.kern.variance.read_value())
print('leng =', model.kern.lengthscales.read_value())


## GP prediction
## =============

# Predict the mean (fmu) and variance (fs2) of the model for the given test data Xtest
fmu, fs2 = model.predict_f(Xtest)
# Add back the mean value that was removed during preprocessing
fmu = fmu + ymean
# Compute bounds of prediction interval
lb = fmu - 2 * np.sqrt(fs2)
ub = fmu + 2 * np.sqrt(fs2)

# Estimate mean and quantiles of the Dirichlet distribution through sampling
q=95 # Set the confidence level
# Initialise arrays to store Dirichlet distribution mean, lower bound, and upper bound
mu_dir = np.zeros([fmu.shape[0], 2])
lb_dir = np.zeros([fmu.shape[0], 2])
ub_dir = np.zeros([fmu.shape[0], 2])
# Generate random samples for Dirichlet distribution estimation
source = np.random.randn(1000, 2)
# Iterate over each predicted sample
for i in range(fmu.shape[0]):
    samples = source * np.sqrt(fs2[i,:]) + fmu[i,:] # Generate Dirichlet distribution samples based on predicted mean and variance
    samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1) # Ensure samples satisfy Dirichlet distribution constraints
    Q = np.percentile(samples, [100-q, q], axis=0) # Compute percentiles to estimate confidence interval
    # Store mean, lower bound, and upper bound in respective arrays
    mu_dir[i,:] = samples.mean(0)
    lb_dir[i,:] = Q[0,:]
    ub_dir[i,:] = Q[1,:]






## Plotting results
## ================

plt.figure(figsize=(12,4))

# to plot the tranformed labels and their standard deviation
Y_tilte += ymean
s1_tilte = np.sqrt(s2_tilte)

plt.subplot(2, 2, 1)
plt.errorbar(X, Y_tilte[:,0], yerr=s1_tilte[:,0], fmt='o', label='Data')
plt.plot(Xtest, fmu[:,0], 'b', label='Posterior')
plt.fill_between(Xtest.flatten(), ub[:,0], lb[:,0], facecolor='0.75')
plt.title('Class 0')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(Xtest, mu_dir[:,0], 'b', label='Prediction')
plt.plot(X, 1-y, 'o', label='Data')
plt.fill_between(Xtest.flatten(), ub_dir[:,0], lb_dir[:,0], facecolor='0.75')
plt.title('Class 0')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 3)
plt.errorbar(X, Y_tilte[:,1], yerr=s1_tilte[:,1], fmt='o', label='Data')
plt.plot(Xtest, fmu[:,1], 'b', label='Posterior')
plt.fill_between(Xtest.flatten(), ub[:,1], lb[:,1], facecolor='0.75')
plt.title('Class 1')
plt.xticks([], [])
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Xtest, mu_dir[:,1], 'b', label='Prediction')
plt.plot(X, y, 'o', label='Data')
plt.fill_between(Xtest.flatten(), ub_dir[:,1], lb_dir[:,1], facecolor='0.75')
plt.title('Class 1')
plt.xticks([], [])
plt.legend()

plt.show()
quit()