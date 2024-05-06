import gpflow
gpflow.kernels.RBF(1) # useless code; just to show any warnings early

import numpy as np
import heteroskedastic 
import matplotlib.pylab as plt


def nll(p, y):
    y = y.astype(int).flatten()
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.flatten()
        P = np.zeros([y.size, 2])
        P[:,0], P[:,1] = 1-p, p
        p = P
    classes = p.shape[1]
    Y = np.zeros((y.size, classes))
    for i in range(y.size):
        Y[i,y[i]] = 1
    logp = np.log(p)
    logp[np.isinf(logp)] = -750
    loglik = np.sum(Y * logp, 1)
    return -np.sum(loglik)

N = 20  # training data
np.random.seed(1235)



## create synthetic dataset
## ===============================
xmax = 15
X = np.random.rand(N,1) * xmax
Xtest = np.linspace(0, xmax*1.5, 200).reshape(-1, 1)
Z = X.copy()

y = np.cos(X.flatten()) / 2 + 0.5
y = np.random.rand(y.size) > y
y = y.astype(int)
if np.sum(y==1) == 0:
    y[0] = 1
elif np.sum(y==0) == 0:
    y[0] = 0


ytest = np.cos(Xtest.flatten()) / 2 + 0.5
ytest = np.random.rand(ytest.size) > ytest
ytest = ytest.astype(int)
if np.sum(ytest==1) == 0:
    ytest[0] = 1
elif np.sum(ytest==0) == 0:
    ytest[0] = 0


# one-hot vector encoding
y_vec = y.astype(int)
classes = np.max(y_vec).astype(int) + 1
Y = np.zeros((len(y_vec), classes))
for i in range(len(y_vec)):
    Y[i, y_vec[i]] = 1
 
a_eps = 0.1

def nll(p, y):
    y = y.astype(int).flatten()
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.flatten()
        P = np.zeros([y.size, 2])
        P[:,0], P[:,1] = 1-p, p
        p = P
    classes = p.shape[1]
    Y = np.zeros((y.size, classes))
    for i in range(y.size):
        Y[i,y[i]] = 1
    logp = np.log(p)
    logp[np.isinf(logp)] = -750
    loglik = np.sum(Y * logp, 1)
    return -np.sum(loglik)



def MNLL_alpha(a_eps):
    s2_tilde = np.log(1.0/(Y+a_eps) + 1)
    Y_tilde = np.log(Y+a_eps) - 0.5 * s2_tilde
    ymean = np.log(Y.mean(0)) + np.mean(Y_tilde-np.log(Y.mean(0)))
    Y_tilde = Y_tilde - ymean


    kernel = gpflow.kernels.RBF(variance=np.var(Y_tilde), lengthscales=np.std(X))
    model = heteroskedastic.SGPRh((X, Y_tilde), kernel=kernel, sn2=s2_tilde, Z=Z)

    opt = gpflow.optimizers.Scipy()
    before = model.maximum_log_likelihood_objective()
    opt.minimize(model.training_loss, model.trainable_variables)
    after = model.maximum_log_likelihood_objective()
    
    fmu, fs2 = model.predict_f(Xtest)
    fmu = fmu + ymean


    gpd_prob = np.zeros(fmu.shape)
    source = np.random.randn(1000, classes)
    for i in range(fmu.shape[0]):
        samples = source * np.sqrt(fs2[i,:]) + fmu[i,:]
        samples = np.exp(samples) / np.exp(samples).sum(1).reshape(-1, 1)
        gpd_prob[i,:] = samples.mean(0)

    gpd_nll = nll(gpd_prob, ytest)

    
    return gpd_nll

alpha = []
MNLL = []
for a_eps in np.linspace(1e-10, 0.2, 20):
    MNLL.append(MNLL_alpha(a_eps))
    alpha.append(a_eps)
    
plt.scatter(alpha, MNLL, marker='o')
plt.xlabel('Alpha')
plt.ylabel('MNLL')
plt.show()
