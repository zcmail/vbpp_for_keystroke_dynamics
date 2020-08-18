# Copyright (C) PROWLER.io 2017
#
# Licensed under the Apache License, Version 2.0

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

import numpy as np
import tensorflow as tf
import gpflow

def tf_calc_Psi_matrix_SqExp(Z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z,z') = ∫ K(z,x) K(x,z') dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).

    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.

    Does not broadcast over leading dimensions.
    """
    variance = tf.cast(variance, Z.dtype)
    lengthscales = tf.cast(lengthscales, Z.dtype)

    mult = tf.cast(0.5 * np.sqrt(np.pi), Z.dtype) * lengthscales      #2分之根号π * lengthscales是根号alpha_r
    inv_lengthscales = 1.0 / lengthscales                             #lengthscales分之一

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    z1 = tf.expand_dims(Z, 1)    # Z增加维度：[M, 1] ==> [M, 1加, 1]
    z2 = tf.expand_dims(Z, 0)    # Z增加维度：[M, M] ==> [1加, M, 1]
    zm = (z1 + z2)/2.0           # Zm = [1/2,1/2,z]

    exp_arg = tf.reduce_sum(
        - tf.square(z1 - z2) / (4.0 * tf.square(lengthscales)),
        axis=2)
    #print(z1)
    #tf.print(z1)
    '''
    sess = tf.InteractiveSession()
    with sess.as_default():
        print('z1 is:',z1.eval())
        print('z2 is:', z2.eval())
        print('zm is:', zm.eval())
        print ('exp_arg is:',exp_arg.eval())
    '''
    erf_val = (tf.math.erf((zm - Tmin) * inv_lengthscales) -
               tf.math.erf((zm - Tmax) * inv_lengthscales))
    product = tf.reduce_prod(mult * erf_val, axis=2)
    Ψ = tf.square(variance) * tf.exp(exp_arg + tf.math.log(product))
    return Ψ

def tf_calc_Psi_matrix(kernel, inducing_var, domain):
    if (isinstance(inducing_var, gpflow.inducing_variables.InducingPoints) and
            isinstance(kernel, gpflow.kernels.SquaredExponential)):
        return tf_calc_Psi_matrix_SqExp(inducing_var.Z, kernel.variance, kernel.lengthscales, domain)
    else:
        raise NotImplementedError("tf_calc_Psi_matrix only implemented for SquaredExponential "
                                  "kernel with InducingPoints")
