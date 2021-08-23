# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:41:39 2021

@author: mohammed batis
"""

import numpy as np
import matplotlib.pyplot as plt

# 根据当前的theta求Y的估计值
# 传入的data_x的最左侧列为全1，即设X_0 = 1
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
      
        for i in range(2):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat    

def return_Y_estimate(theta_now, data_x):
    # 确保theta_now为列向量
    theta_now = theta_now.reshape(-1, 1)
    _Y_estimate = np.dot(data_x, theta_now)

    return _Y_estimate


# 求当前theta的梯度
# 传入的data_x的最左侧列为全1，即设X_0 = 1
def return_dJ(theta_now, data_x, y_true):
    y_estimate = return_Y_estimate(theta_now, data_x)
    # 共有_N组数据
    _N = data_x.shape[0]
    # 求解的theta个数
    _num_of_features = data_x.shape[1]
    # 构建
    _dJ = np.zeros([_num_of_features, 1])
    
    for i in range(_num_of_features):
        _dJ[i, 0] = 2 * np.dot((y_estimate - y_true).T, data_x[:, i]) / _N
    
    return _dJ


# 计算J的值
# 传入的data_x的最左侧列为全1，即设X_0 = 1
def return_J(theta_now, data_x, y_true):
    # 共有N组数据
    N = data_x.shape[0]
    temp = y_true - np.dot(data_x, theta_now)
    _J = np.dot(temp.T, temp) / N
    
    return _J


# 梯度下降法求解线性回归
# data_x的一行为一组数据
# data_y为列向量，每一行对应data_x一行的计算结果
# 学习率默认为0.3
# 误差默认为1e-8
# 默认最大迭代次数为1e4
def gradient_descent(data_x, data_y, Learning_rate = 0.01, ER = 1e-5, MAX_LOOP = 50):
    # 样本个数为
    _num_of_samples = data_x.shape[0]
    # 在data_x的最左侧拼接全1列
    X_0 = np.ones([_num_of_samples, 1])
    new_x = np.column_stack((X_0, data_x))
    print(new_x)
    # 确保data_y为列向量
    new_y = data_y.reshape(-1, 1)
    print(new_y)
    # 求解的未知元个数为
    _num_of_features = new_x.shape[1]
    # 初始化theta向量
    theta = np.zeros([_num_of_features, 1]) * 0.0
    flag = 0  	# 定义跳出标志位
    last_J = 0  # 用来存放上一次的Lose Function的值
    ct = 0  	# 用来计算迭代次数
    
    while flag == 0 and ct < MAX_LOOP:
        last_theta = theta
        # 更新theta
        gradient =  return_dJ(theta, new_x, new_y)
        
        theta = theta - Learning_rate * gradient
        
        er = abs(return_J(last_theta, new_x, new_y) - return_J(theta, new_x, new_y))
        
        # 误差达到阀值则刷新跳出标志位
        if er < ER :
            flag = 1
        
        # 叠加迭代次数
        ct += 1
   
    return theta


def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error
   
if __name__ == '__main__':
    xArr,yArr=loadDataSet('test.txt')
    # print(gradient_descent(np.array(xArr),np.array(yArr)))
   
    theta = gradient_descent(np.array(xArr), np.array(yArr))
    cost,error = cost_function(np.array(xArr), np.array(yArr),theta)
    print(cost,error)
    # print(theta)

    