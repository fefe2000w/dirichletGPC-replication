# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:14:15 2024

@author: lisy
"""

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.models import GPModel
from gpflow.likelihoods import Likelihood



# Defines a class GaussianHeteroskedastic, which is a subclass of Likelihood, representing a likelihood function for heteroskedastic Gaussian distribution
class GaussianHeteroskedastic(Likelihood):
    def 