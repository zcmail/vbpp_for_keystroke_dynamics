# Copyright (C) PROWLER.io 2017-2019
#
# Licensed under the Apache License, Version 2.0

# run from this file's directory

import numpy as np
import matplotlib.pyplot as plt
import gpflow

from vbpp.model import VBPP
from data_up_events_training import make_estimate_data

def build_data(user, start_line, end_line, filename):
    #events_oral = make_estimate_data(user,start_line,end_line,filename)
    #events = np.unique(events_oral.flatten())
    events = make_estimate_data(user, start_line, end_line, filename)
    num_observations = len(events)
    #print(num_observations)
    return events, num_observations

def domain_grid(domain, num_points):       #域grid
    return np.linspace(domain.min(axis=1), domain.max(axis=1), num_points)

def domain_area(domain):                   #域面积
    return np.prod(domain.max(axis=1) - domain.min(axis=1))

def build_model(events, domain, num_observations, M=20):
    #kernel = gpflow.kernels.SquaredExponential()
    kernel = gpflow.kernels.SquaredExponential(variance = 1.0, lengthscales = 0.5)
    Z = domain_grid(domain, M)                               #均匀切分domain,和events无关
    feature = gpflow.inducing_variables.InducingPoints(Z)    #inducing point（将均匀切分的点作为inducing point）
    q_mu = np.zeros(M)      #均值为0？
    q_S = np.eye(M)         #单位矩阵
    #print (events)
    num_events = len(events)
    beta0 = np.sqrt(num_events / domain_area(domain))       # 事件数/域面积 的开方,是什么？  是第二个模型的offset
    model = VBPP(feature, kernel, domain, q_mu, q_S, beta0=beta0, num_events=num_events, num_observations = num_observations)
    return model

def demo():
    N = 100     #预测点（lambda）
    #目标用户信息,用于模型训练
    object_user = 's036'
    object_data_str = 0
    object_data_end = 200
    #测试用户数据
    test_uset = 's002'
    test_data_str = 0
    test_data_end = 400

    filename = "./data/DSL-StrongPasswordData.xls"
    events,num_observations = build_data(object_user,object_data_str, object_data_end,filename)
    events = np.array(events, float).reshape(-1, 1)

    domain_max = max(events) + 0.03
    domain = [0,domain_max]
    domain = np.array(domain, float).reshape(1, 2)

    model = build_model(events, domain, num_observations, M=4)   #M是 inducing point的数量

    def objective_closure():                           #目标函数
        return - model.elbo(events)

    gpflow.optimizers.Scipy().minimize(objective_closure, model.trainable_variables)

    #画强度的估值图
    X = domain_grid(domain, N)
    lambda_mean, lower, upper = model.predict_lambda_and_percentiles(X)
    lower = lower.numpy().flatten()
    upper = upper.numpy().flatten()
    plt.subplot(1, 2, 1)
    plt.xlim(X.min(), X.max())
    #plt.ylim(0, 15)
    plt.plot(X, lambda_mean)
    plt.fill_between(X.flatten(), lower, upper, alpha=0.3)
    plt.plot(events, np.zeros_like(events), '|')
    #plt.show()

    #画预测图
    test_data_all = []
    for i in range(0,test_data_end):     #预测值显示
        events_for_test, _ = build_data(test_uset, test_data_str+i, test_data_str + i + 1, filename)
        events_for_test = np.array(events_for_test, float).reshape(-1, 1)
        test_data_likelihood = model.predict_y(events_for_test, domain, N)
        test_data_all.append(test_data_likelihood)
        print('test num is:',i)
    #print (test_data_likelihood)
    plt.subplot(1, 2, 2)
    x_aix = range(test_data_end)
    #plt.ylim(-50, 2)
    plt.plot(x_aix, test_data_all)
    plt.show()

if __name__ == "__main__":
    demo()