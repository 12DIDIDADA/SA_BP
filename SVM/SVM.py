# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv',encoding=  'gbk')
x1 = data[['F1', 'F2', 'F3', 'F4', '地层', '土地利用', '土壤类型']]
y = data['As']

scaler = StandardScaler()
x = scaler.fit_transform(x1)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=88)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1/2, random_state=88)
# 创建 SVM 分类模型
svm_model = SVR(kernel='rbf',#核函数类型
                C=100, #正则化参数
                # degree='2 #多项式核函数的次数支队poly有效',
                epsilon = 0.1,#误差容忍度
                gamma = 0.1,#核函数系数
                max_iter= 1000,
                shrinking= True,#是否使用收缩启发式
                tol = 0.001,#优化算法的终止条件
                )

# 训练模型
svm_model.fit(x_train, y_train)

y_test_pred = svm_model.predict(x_test)

# 计算 R²、MSE 和 MAE
r2 = r2_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
print(f'\nSVM模型的验证集评估结果:')
print(f'R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}')
data_test = pd.DataFrame()
data_test['y_test'] = y_test
data_test['y_test_pred'] = y_test_pred
data_test.to_csv(r'/root/SVM/svm_test.csv', index=False)

y_val_pred = svm_model.predict(x_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)

print(f'\nsvm模型的测试集评估结果:')
print(f'R²: {r2_val:.2f}')
print(f'MSE: {mse_val:.4f}')
print(f'MAE: {mae_val:.3f}')



