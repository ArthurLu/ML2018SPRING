
# coding: utf-8

# In[1]:
import sys
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logistic_fun(value):
    return 1.0/(1.0+np.exp(-value))  # paralleled, work with ndarray

def logistic_regression_with_GD(data, label, sita, times, initial=None, plot=False, data_test=None, label_test=None):
    if initial==None:
        parameter = np.random.rand(data.shape[1])
    else:
        parameter = initial
        
    
    parameter_list = []
    parameter_list.append(list(parameter))
    E_in = []
    E_out = []
    
    for i in range(times):
        # direction
        temp1 = np.dot(data, parameter)                         # WT*xn，矩阵向量乘得到 X*w vector
        temp2 = np.multiply(label, temp1)                       # yn*WT*xn，向量对应位置相乘得到 Y.*(X*w) vector
        logistic_vector = logistic_fun( -temp2 )                # logistic_fun vector，向量对应位置相乘得到 s(.) vector
        scalar_vector = np.multiply(logistic_vector, -label)    # scalar vector，向量对应位置相乘得到 s(.).*(-Y) vector
        gradient_direction = np.dot(scalar_vector, data)/len(label) # gradient direction，横的向量左乘矩阵 = 向量每个元素乘矩阵一行再相加，即同时完成标量向量乘和最后的求和，得到vector

        # iterate
        parameter -= sita*gradient_direction
        parameter_list.append(list(parameter))  # return all parameter vectors(ndarry) as a list

    return parameter_list


def logistic_regression_with_SGD(data, label, sita, times, sequence=None):
    parameter = np.random.rand(data.shape[1])
    parameter_list = []
    parameter_list.append(list(parameter))
    
    for i in range(times):
        # direction
        index = sequence[i]
        xi = data[index, :]
        yi = label[index]
        scalar = logistic_fun(-yi*np.dot(parameter, xi))*(-yi)
        stochastic_gradient_direction = scalar*xi
        
        # iterate
        parameter -= sita*stochastic_gradient_direction
        parameter_list.append(list(parameter))  # return all parameter vectors(ndarry) as a list

    return parameter_list    


def E_01(parameter, data, label):
    s = np.dot(data, parameter)
    forecast = np.sign(s)
    err = sum(abs((forecast-label)/2))/len(label)
    err = float(err)

    return err


# In[2]:

## Part 1: read file and extract data
train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
prediction_y_path = sys.argv[4]

train_x = np.loadtxt(train_x_path, delimiter=',', skiprows=1)
train_y = np.loadtxt(train_y_path)
test_x = np.loadtxt(test_x_path, delimiter=',', skiprows=1)


# train data
train_dimension = train_x.shape
train_N = train_dimension[0]
train_d = train_dimension[1]  # 不包括自己加的常数项
#print(train_dimension)



# test data
test_dimension = test_x.shape
test_N = test_dimension[0]
test_d = test_dimension[1]
#print(test_dimension)



# In[17]:

## normalized


def normalized(data, specific_columns=None):
    
    if specific_columns == None:
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        
        data_normalized = np.zeros((data.shape[0], data.shape[1]))
        
        for i in range(data.shape[0]):
            data_normalized[i] = (data[i]-data_min)/(data_max-data_min)
            
    
    else:
        
        data_normalized = data.copy()
        
        for col in specific_columns:
            col_max = np.max(data[:, col])
            col_min = np.min(data[:, col])
            
            data_normalized[:, col] = (data_normalized[:, col]-col_min)/(col_max-col_min)
            
        
    data_normalized = np.column_stack( (np.ones(data.shape[0]), data_normalized) )
        
    return data_normalized

# train data
continuous_columns = [0, 10, 78, 79, 80]

train_x_normalized = normalized(train_x, specific_columns=continuous_columns)

train_y_transformed = np.ones(train_N)

for i in range(train_N):
    if train_y[i]:
        train_y_transformed[i] = 1.
    else:
        train_y_transformed[i] = -1.

# test data
test_x_normalized = normalized(test_x, specific_columns=continuous_columns)




# In[27]:

## Part 2: logistic regression with GD
sita = 0.01
T = 70000

w = logistic_regression_with_GD(train_x_normalized, train_y_transformed, sita, T, initial=None)



# In[31]:

def predict(w, x):
    s = np.dot(x, w)
    forecast = np.sign(s)
    
    
    forecast_transformed = np.where(forecast==-1, 0, 1)
    
    return forecast_transformed, forecast

f1, f2 = predict(w[-1], test_x_normalized)





# In[32]:
index = []

for i in range(len(f1)):
    index.append(i+1)

result = pd.DataFrame({'id':index, 'label':f1})
result.to_csv(prediction_y_path, index=False)

