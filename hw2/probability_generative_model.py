
# coding: utf-8

# In[44]:
import sys
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc


# In[6]:

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


# In[14]:



# 分类data
amount_of_class1 = int(sum(train_y))
#print(amount_of_class1)

amount_of_class2 = train_N - amount_of_class1
#print(amount_of_class2)

train_x_class1 = np.zeros((amount_of_class1, train_d)) # label:1
train_x_class2 = np.zeros((amount_of_class2, train_d)) # label:0

index1 = 0
index2 = 0

mapping_index_class1 = []
mapping_index_class2 = []

for i in range(train_N):
    if train_y[i] == 1:
        train_x_class1[index1]=train_x[i]
        mapping_index_class1.append(i)
        index1 += 1
        
        
    elif train_y[i] == 0:
        train_x_class2[index2]=train_x[i]
        mapping_index_class2.append(i)
        index2 += 1
    
    else:
        raise ValueError
    





# In[50]:

## 混合正态分布（连续型特征）和两点分布（离散型特征）

# 总体分类概率
total_pro_class1 = amount_of_class1/train_N
total_pro_class2 = amount_of_class2/train_N

# fit 正太分布
continuous_columns = [0, 10, 78, 79, 80]
con_d = len(continuous_columns)
con_data_class1 = np.zeros((amount_of_class1, con_d))
con_data_class2 = np.zeros((amount_of_class2, con_d))

col_index = 0

for col in continuous_columns:
    con_data_class1[:, col_index] = train_x_class1[:, col]
    con_data_class2[:, col_index] = train_x_class2[:, col]
    col_index += 1
    
    
con_mean_class1 = np.mean(con_data_class1, axis=0)
con_mean_class2 = np.mean(con_data_class2, axis=0)

con_cov_class1 = np.cov(con_data_class1.T, bias=True)
con_cov_class2 = np.cov(con_data_class2.T, bias=True)

con_inv_cov_class1 = np.mat(con_cov_class1).I
con_inv_cov_class2 = np.mat(con_cov_class2).I

con_det_cov_class1 = np.linalg.det(con_cov_class1)
con_det_cov_class2 = np.linalg.det(con_cov_class2)

# fit 二项分布
bornuli_class1 = [] # 只保存成功概率
bornuli_class2 = []

for i in range(train_d):
    if i not in continuous_columns:
        bornuli_class1.append(sum(train_x_class1[:, i])/amount_of_class1)
        bornuli_class2.append(sum(train_x_class2[:, i])/amount_of_class2)


# In[99]:

# predict

# 定义一个正态分布的计算函数

def normal_distribution(x, u, sigma_inv, sigma_det):
    D = len(u)
    
    temp1 = x-u
    temp2 = np.dot(temp1.T, sigma_inv)
    temp3 = np.dot(temp2, temp1)
    temp4 = np.exp(-temp3/2)
    temp5 = temp4/(math.sqrt(sigma_det)*(math.pi**(D/2)))
    
    #print(temp3, temp4, temp5)
    
    return temp5


# 提取连续型数据
con_data_test = np.zeros((test_N, con_d))

con_prob_class1 = np.zeros(test_N)
con_prob_class2 = np.zeros(test_N)

disc_prob_class1 = np.zeros(test_N)
disc_prob_class2 = np.zeros(test_N)

prob_class1 = np.zeros(test_N)
prob_class2 = np.zeros(test_N)

condition_prob_class1 = np.zeros(test_N)
condition_prob_class2 = np.zeros(test_N)

prediction = []

col_index = 0
bornuli_index = 0

for col in continuous_columns:
    con_data_test[:, col_index] = test_x[:, col]
    col_index += 1
    
    
# 逐个data预测
for i in range(test_N):
    
    con_prob_class1[i] = normal_distribution(con_data_test[i], con_mean_class1, con_inv_cov_class1, con_det_cov_class1)
    con_prob_class2[i] = normal_distribution(con_data_test[i], con_mean_class2, con_inv_cov_class2, con_det_cov_class2)
                   
        
    bornuli_index = 0
    for j in range(test_d):
        
        if j not in continuous_columns:
            
            if test_x[i, j] == 1:
                disc_prob_class1 += bornuli_class1[bornuli_index]
                disc_prob_class2 += bornuli_class2[bornuli_index]
                bornuli_index += 1
                
            elif test_x[i, j] == 0:
                disc_prob_class1 += (1.0-bornuli_class1[bornuli_index])
                disc_prob_class2 += (1.0-bornuli_class2[bornuli_index])
                bornuli_index += 1
                
            else:
                #print('error')
                raise ValueError
                
    
    prob_class1[i] = (con_prob_class1[i]+disc_prob_class1[i])*total_pro_class1               
    prob_class2[i] = (con_prob_class2[i]+disc_prob_class2[i])*total_pro_class2
    
    condition_prob_class1[i] = prob_class1[i]/(prob_class1[i]+prob_class2[i])
    condition_prob_class2[i] = prob_class2[i]/(prob_class1[i]+prob_class2[i])
    
    
    if condition_prob_class1[i]>condition_prob_class2[i]:
        prediction.append(1)
    else:
        prediction.append(0)


# In[101]:

index = []
for i in range(len(prediction)):
    index.append(i+1)
    
result = pd.DataFrame({'id':index, 'label':prediction})

result.to_csv(prediction_y_path, index=False)




