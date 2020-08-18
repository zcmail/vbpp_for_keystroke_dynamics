# Copyright (C) PROWLER.io 2017-2019
#
# Licensed under the Apache License, Version 2.0

"""
Prototype Code! This code may not be fully tested, or in other ways fit-for-purpose.
Use at your own risk!
"""

from typing import Optional
import gpflow
import numpy as np
import tensorflow as tf
from functools import reduce
from gpflow import Parameter
from gpflow import kullback_leiblers
from gpflow.config import default_float
from gpflow.utilities import positive, triangular, to_default_float
from gpflow.conditionals import conditional
from scipy.stats import ncx2
from scipy import integrate

from .tf_utils import tf_len, tf_vec_mat_vec_mul, tf_vec_dot, tf_squeeze_1d
from .Gtilde import tf_Gtilde_lookup
from .psimatrix import tf_calc_Psi_matrix
from .phivector import tf_calc_Phi_vector


def _integrate_log_fn_sqr(mean, var):
    """
    ∫ log(f²) N(f; μ, σ²) df  from -∞ to ∞
    """
    z = - 0.5 * tf.square(mean) / var
    C = 0.57721566  # Euler-Mascheroni constant
    G = tf_Gtilde_lookup(z)
    return - G + tf.math.log(0.5 * var) - C

def integrate_log_fn_sqr(mean, var):
    # N = tf_len(μn)
    #μn = tf_squeeze_1d(mean)
    #σ2n = tf_squeeze_1d(var)
    integrated = _integrate_log_fn_sqr(mean, var)
    point_eval = tf.math.log(mean ** 2)  # TODO use mvn quad instead?
    # TODO explain
    return tf.where(tf.math.is_nan(integrated), point_eval, integrated)

class VBPP(gpflow.models.GPModel):
    """
    Implementation of the "Variational Bayes for Point Processes" model by
    Lloyd et al. (2015), with capability for multiple observations and the
    constant offset `beta0` from John and Hensman (2018).
    """

    def __init__(self,
                 inducing_variable: gpflow.inducing_variables.InducingVariables,
                 kernel: gpflow.kernels.Kernel,
                 domain: np.ndarray,
                 q_mu: np.ndarray,
                 q_S: np.ndarray,
                 *,
                 beta0: float = 1e-6,
                 num_observations: int = 1,
                 num_events: Optional[int] = None,
                 num_latent_gps: int = 1,     #？潜在高斯过程数，这个变量后面没有起作用
                 ):
        """
        D = number of dimensions                                       维度
        M = size of inducing variables (number of inducing points)     inducing变量数

        :param inducing_variable: inducing variables (here only implemented for a gpflow      inducing变量
            .inducing_variables.InducingPoints instance, with Z of shape M x D)
        :param kernel: the kernel (here only implemented for a gpflow.kernels                 kernel（这里只有高斯核）
            .SquaredExponential instance)
        :param domain: lower and upper bounds of (hyper-rectangular) domain                   domain（domain的上界和下界）
            (D x 2)

        :param q_mu: initial mean vector of the variational distribution q(u)                 q(u)的均值向量
            (length M)
        :param q_S: how to initialise the covariance matrix of the variational                q(u)的协方差矩阵
            distribution q(u)  (M x M)

        :param beta0: a constant offset, corresponding to initial value of the                应该大，以至于GP不会为负？
            prior mean of the GP (but trainable); should be sufficiently large
            so that the GP does not go negative...

        :param num_observations: number of observations of sets of events                     观察到的事件集集数？
            under the distribution

        :param num_events: total number of events, defaults to events.shape[0]                事件总数
            (relevant when feeding in minibatches)
        """
        super().__init__(kernel, likelihood=None, num_latent_gps=num_latent_gps)  # custom likelihood

        # observation domain  (D x 2)
        self.domain = domain
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise ValueError("domain must be of shape D x 2")

        self.num_observations = num_observations
        self.num_events = num_events

        if not (isinstance(kernel, gpflow.kernels.SquaredExponential)
                and isinstance(inducing_variable, gpflow.inducing_variables.InducingPoints)):
            raise NotImplementedError("This VBPP implementation can only handle real-space "
                                      "inducing points together with the SquaredExponential "
                                      "kernel.")
        self.kernel = kernel
        self.inducing_variable = inducing_variable

        self.beta0 = Parameter(beta0, transform=positive(), name="beta0")  # constant mean offset  GPflow超参数优化

        # variational approximate Gaussian posterior q(u) = N(u; m, S)
        self.q_mu = Parameter(q_mu, name="q_mu")  # mean vector  (length M)

        # covariance:
        L = np.linalg.cholesky(q_S)  # S = L L^T, with L lower-triangular  (M x M)      #Cholesky分解
        self.q_sqrt = Parameter(L, transform=triangular(), name="q_sqrt")

        self.psi_jitter = 0.0                        #Ψ偏移

    def _Psi_matrix(self):
        Ψ = tf_calc_Psi_matrix(self.kernel, self.inducing_variable, self.domain)
        psi_jitter_matrix = self.psi_jitter * tf.eye(len(self.inducing_variable), dtype=default_float())
        return Ψ + psi_jitter_matrix 

    @property                                                   #把一个方法变成属性调用
    def total_area(self):
        return np.prod(self.domain[:, 1] - self.domain[:, 0])

    def predict_f(self, Xnew, full_cov=False, *, Kuu=None):     #q(f)的近似
        """
        VBPP-specific conditional on the approximate posterior q(u), including a
        constant mean function.
        """
        mean, var = conditional(Xnew, self.inducing_variable, self.kernel, self.q_mu[:, None],
                                full_cov=full_cov, q_sqrt=self.q_sqrt[None, :, :])
        # TODO make conditional() use Kuu if available
        return mean + self.beta0, var

    def _elbo_data_term(self, events, Kuu=None):                     # E_q [log f_n], log f_n^2 的期望
        #print('len of events',len(events))
        mean, var = self.predict_f(events, full_cov=False, Kuu=Kuu)
        expect_log_fn_sqr = integrate_log_fn_sqr(mean, var)
        if self.num_events is None:
            scale = 1.0
        else:
            minibatch_size = tf.shape(events)[0]
            #tf.print('num_events is', self.num_events)
            #tf.print('minibatch_size is',minibatch_size)
            scale = to_default_float(self.num_events) / to_default_float(minibatch_size)
        return scale * tf.reduce_sum(expect_log_fn_sqr)    #计算张量沿着某一维度的和，默认计算所有元素的和。

    def _var_fx_kxx_term(self):
        if isinstance(self.kernel, gpflow.kernels.Product):    #如果是乘积核
            γ = reduce(lambda a, b: a * b, [k.variance for k in self.kernel.kernels])
        elif isinstance(self.kernel, gpflow.kernels.Sum):      #如果是求和核
            γ = reduce(lambda a, b: a + b, [k.variance for k in self.kernel.kernels])
        else:
            γ = self.kernel.variance
        kxx_term = γ * self.total_area
        return kxx_term

    def _elbo_integral_term(self, Kuu):               #lamda 的积分
        """
        Kuu : dense matrix
        """
        # q(f) = GP(f; μ, Σ)
        Ψ = self._Psi_matrix()

        # int_expect_fx_sqr = m^T Kzz⁻¹ Ψ Kzz⁻¹ m
        # = (Kzz⁻¹ m)^T Ψ (Kzz⁻¹ m)

        # Kzz = R R^T
        R = tf.linalg.cholesky(Kuu)

        # Kzz⁻¹ m = R^-T R⁻¹ m
        # Rinv_m = R⁻¹ m
        Rinv_m = tf.linalg.triangular_solve(R, self.q_mu[:, None], lower=True)

        # R⁻¹ Ψ R^-T
        # = (R⁻¹ Ψ) R^-T
        Rinv_Ψ = tf.linalg.triangular_solve(R, Ψ, lower=True)
        # = (Rinv_Ψ) R^-T = (R⁻¹ Rinv_Ψ^T)^T
        Rinv_Ψ_RinvT = tf.linalg.triangular_solve(R, tf.transpose(Rinv_Ψ), lower=True)

        int_mean_f_sqr = tf_vec_mat_vec_mul(Rinv_m, Rinv_Ψ_RinvT, Rinv_m)

        Rinv_L = tf.linalg.triangular_solve(R, self.q_sqrt, lower=True)
        Rinv_L_LT_RinvT = tf.matmul(Rinv_L, Rinv_L, transpose_b=True)

        # int_var_fx = γ |T| + trace_terms
        # trace_terms = - Tr(Kzz⁻¹ Ψ) + Tr(Kzz⁻¹ S Kzz⁻¹ Ψ)
        trace_terms = tf.reduce_sum((Rinv_L_LT_RinvT - tf.eye(len(self.inducing_variable), dtype=default_float())) *
                                    Rinv_Ψ_RinvT)

        kxx_term = self._var_fx_kxx_term()
        int_var_f = kxx_term + trace_terms

        f_term = int_mean_f_sqr + int_var_f

        # λ = E_f{(f + β₀)**2}
        #   = (E_f)^2 + var_f + 2 f β₀ + β₀^2
        #   = f_term + int_cross_terms + betas_term
        Kuu_inv_m = tf.linalg.triangular_solve(tf.transpose(R), Rinv_m, lower=False)

        Phi = tf_calc_Phi_vector(self.kernel, self.inducing_variable, self.domain)   #计算Ψ
        int_cross_term = 2 * self.beta0 * tf_vec_dot(Phi, Kuu_inv_m)

        beta_term = tf.square(self.beta0) * self.total_area

        int_lambda = f_term + int_cross_term + beta_term    #fn 的似然

        return - int_lambda

    def prior_kl(self, Kuu):
        """
        KL divergence between p(u) = N(0, Kuu) and q(u) = N(μ, S)
        """
        return kullback_leiblers.gauss_kl(self.q_mu[:, None], self.q_sqrt[None, :, :], Kuu)

    def elbo(self, events):                                #log Likelihood log p(D|theta)
        """
        Evidence Lower Bound (ELBo) for the log likelihood.
        """
        K = gpflow.covariances.Kuu(self.inducing_variable, self.kernel)

        integral_term = self._elbo_integral_term(K)
        data_term = self._elbo_data_term(events, K)
        kl_div = self.prior_kl(K)

        #elbo = integral_term + data_term - kl_div
        #tf.print('elbo is:', elbo)
        elbo = self.num_observations * integral_term + data_term - kl_div
        return elbo

    def compute_Kuu(self):              #这个在这没有被调用，计算Kuu？
        return gpflow.covariances.Kuu(self.inducing_variable, self.kernel)

    def predict_lambda(self, Xnew):       #强度函数的期望？
        """
        Expectation value of the rate function of the Poisson process.

        :param xx: points at which to calculate
        """
        mean_f, var_f = self.predict_f(Xnew)
        λ = tf.square(mean_f) + var_f
        return λ

    def predict_lambda_and_percentiles(self, Xnew, lower=5, upper=95):    #强度函数的期望，百分位上下界
        """
        Computes mean value of intensity and lower and upper percentiles.
        `lower` and `upper` must be between 0 and 100.
        """
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(Xnew)
        # λ = E[f²] = E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f
        # g = f/√var_f ~ Normal(mean_f/√var_f, 1)
        # g² = f²/var_f ~ χ²(k=1, λ=mean_f²/var_f) non-central chi-squared
        m2ov = mean_f ** 2 / var_f
        if tf.reduce_any(m2ov > 10e3):
            raise ValueError("scipy.stats.ncx2.ppf() flatlines for nc > 10e3")
        f2ov_lower = ncx2.ppf(lower/100, df=1, nc=m2ov)
        f2ov_upper = ncx2.ppf(upper/100, df=1, nc=m2ov)
        # f² = g² * var_f
        lambda_lower = f2ov_lower * var_f
        lambda_upper = f2ov_upper * var_f
        return lambda_mean, lambda_lower, lambda_upper

    def predict_y(self, Xnew, domain, M):                             #基于lambda 数据的log_likelihood
        #cal_term
        X = np.linspace(domain.min(axis=1), domain.max(axis=1), M)
        mean_f,var_f = self.predict_f(X)
        lambda_mean = (mean_f) ** 2 + var_f
        lambda_pre = np.array(lambda_mean).reshape(M,)
        X = X.reshape(M,)
        cal_term = integrate.trapz(lambda_pre, X)
        #print('lambda_mean is:', lambda_pre, lambda_pre.shape,type(lambda_pre))
        #print('X is:', X, X.shape,type(X))
        #print('cal_term is:', cal_term)
        mean_pre, var_pre = self.predict_f(Xnew)                               #在test数据上的lambda估计值
        #data_term
        #print('mean_f is',mean_pre)
        #data_term = tf.reduce_sum((mean_pre)**2 + var_pre)                    #修正Jhon的小错误，加上log
        data_term = tf.reduce_sum(np.log((mean_pre) ** 2 + var_pre))
        #print('data_term is:', data_term)
        #print('data_term is:', data_term)
        test_data_likelihood = data_term - cal_term
        #print('cal_term is',cal_term)
        return test_data_likelihood

    def predict_density(self, new_events):                 #未用
        raise NotImplementedError
