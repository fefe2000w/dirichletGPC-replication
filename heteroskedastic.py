import numpy as np
import tensorflow as tf
from check_shapes import inherit_check_shapes

import gpflow
from gpflow.models import GPModel
from gpflow.likelihoods import ScalarLikelihood

from gpflow import logdensities
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper


class GaussianHeteroskedastic(ScalarLikelihood):
    def __init__(self, variance=1.0, scale=None, variance_lower_bound=None, **kwargs):
        super().__init__(**kwargs)
        if np.isscalar(variance):
            variance = np.array(variance)
        self.variance = gpflow.Parameter(variance, trainable=False)
        #variance.trainable = False
        self.variance_numel = variance.size
        self.variance_ndim = variance.ndim
        
    @inherit_check_shapes
    def _scalar_log_prob(self, F, Y, X=None):
        return logdensities.gaussian(Y, F, self.variance)
    
    @inherit_check_shapes
    def _conditional_mean(self, F, X=None):
        return tf.idenity(F)
    
    @inherit_check_shapes
    def _conditional_variance(self, F, X=None):
        return tf.broadcast_to(self.variance, tf.shape(F))
    
    @inherit_check_shapes
    def _predict_mean_and_var(self, Fmu, Fvar, X=None):
        return tf.identity(Fmu), Fvar + self.variance
    
    @inherit_check_shapes
    def _predict_log_density(self, Fmu, Fvar, Y, X=None):
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar+self.variance), axis=-1)
    
    @inherit_check_shapes
    def _variational_expectations(self, Fmu, Fvar, Y, X=None):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y-Fmu) ** 2 + Fvar) / self.variance
            )


class SGPRh(GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, sn2, Z, mean_function=None, **kwargs):
        X, Y = data_input_to_tensor(data)
        likelihood = GaussianHeteroskedastic(sn2)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y.shape[-1], **kwargs)
        self.data = X, Y
        self.Z = gpflow.Parameter(Z, trainable=False)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.inducing_variable: InducingPoints = inducingpoint_wrapper(self.Z)
        self.mean_function = mean_function or gpflow.mean_functions.Zero()
    
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self):
        return self.elbo()
    
    def elbo(self):
        X, Y = self.data
        Z = self.inducing_variable
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        num_inducing = self.inducing_variable.num_inducing
        
        #sigma_sq - tf.squeeze()
        kdiag = self.kernel(X, full_cov=False)
        kuf = Kuf(Z, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        invL_kuf = tf.linalg.triangular_solve(L, kuf, lower=True)
        Err = Y - self.mean_function(X)
        
        bound = 0
        for i in range(self.num_latent):
            err = tf.slice(Err, [0, i], [self.num_data, 1])
            
            if self.likelihood.variance_ndim > 1:
                sn2 = self.likelihood.variance[:, i]
            else:
                sn2 = self.likelihood.variance
            sigma = tf.sqrt(sn2)
            
            A = invL_kuf / sigma
            AAT = tf.linalg.matmul(A, A, transpose_b=True)
            B = AAT + tf.eye(num_inducing, dtype=default_float())
            LB = tf.linalg.cholesky(B)
            err_sigma = tf.reshape(err, [self.num_data]) / sigma
            err_sigma = tf.reshape(err_sigma, [self.num_data, 1])
            Aerr = tf.linalg.matmul(A, err_sigma)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
            
            if self.likelihood.variance_numel == 1:
                sum_log_sn2 = num_data * tf.math.log(sn2)
            else:
                sum_log_sn2 = tf.reduce_sum(tf.math.log(sn2))
            
            bound += -0.5 * num_data * np.log(2 * np.pi)
            bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
            bound -= 0.5 * sum_log_sn2
            bound += -0.5 * tf.reduce_sum(tf.square(err_sigma))
            bound += 0.5 * tf.reduce_sum(tf.square(c))
            bound += -0.5 * tf.reduce_sum(kdiag / sn2)
            bound += 0.5 * tf.reduce_sum(tf.linalg.diag_part(AAT))
            
        return bound
    
    @inherit_check_shapes
    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        X, Y = self.data
        Z = self.inducing_variable
        num_inducing = self.inducing_variable.num_inducing
        kuf = Kuf(Z, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        kus = Kuf(Z, self.kernel, Xnew)
        L = tf.linalg.cholesky(kuu)
        invL_kuf = tf.linalg.triangular_solve(L, kuf, lower=True)
        Err = Y - self.mean_function(X)
        
        mu = None
        cov = None
        for i in range(self.num_latent):
            err = tf.slice(Err, [0, i], [self.num_data, 1])
            
            if self.likelihood.variance_ndim > 1:
                sn2 = self.likelihood.variance[:, i]
            else:
                sn2 = self.likelihood.variance
            sigma = tf.sqrt(sn2)
        
            A = invL_kuf / sigma
            AAT = tf.linalg.matmul(A, A, transpose_b=True)
            B = AAT + tf.eye(num_inducing, dtype=default_float())
            LB = tf.linalg.cholesky(B)
            err_sigma = tf.reshape(err, [self.num_data]) / sigma
            err_sigma = tf.reshape(err_sigma, [self.num_data, 1])
            Aerr = tf.linalg.matmul(A, err_sigma)
            c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
            tmp1 = tf.linalg.triangular_solve(L, kus, lower=True)
            tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
            mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        
            if full_cov:
                raise Exception('full_cov not imploemented!')
            else:
                var = self.kernel(Xnew, full_cov=False) + tf.reduce_sum(tf.square(tmp2), 0) \
                    - tf.reduce_sum(tf.square(tmp1), 0)
                shape = tf.stack([1, tf.shape(err)[1]])
                var = tf.tile(tf.expand_dims(var, 1), shape)
        
            if mu is None or cov is None:
                mu = mean
                cov = var
            else:
                mu = tf.concat([mu, mean], 1)
                cov = tf.concat([cov, var], 1)
        
        mu = mu + self.mean_function(Xnew)
        return mu, cov
    
    
    
