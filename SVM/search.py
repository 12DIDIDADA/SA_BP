# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv',encoding=  'gbk')
# data = pd.read_csv(r"D:\arcpy_io\arcmap_data\PCA\ms_pca_fanzhuan.csv", encoding='gbk')

x1 = data[['F1', 'F2', 'F3', 'F4', '地层', '土地利用', '土壤类型']]
y = data['As']

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x1)

# 1. 数据集划分：70% 训练集，30% 临时集（验证集 + 测试集）
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=88)

# 2. 将临时集再次划分为验证集（20%）和测试集（10%）
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1/2, random_state=88)


# 创建SVR模型
svr = SVR()

# 定义参数网格
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],  # 增加poly和sigmoid核函数
    'C': [0.01, 0.1, 1, 10, 100],                   # 扩展C的范围
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],  # 增加gamma选项
    'epsilon': [0.001, 0.01, 0.1, 0.2, 0.5],        # 增加epsilon选项
    'degree': [2, 3, 4, 5],                         # 多项式核的阶数
    'coef0': [0.0, 0.1, 1.0],                       # 核函数中的独立项
    'shrinking': [True, False],                     # 是否使用收缩启发式
    'tol': [1e-3, 1e-4, 1e-5],                      # 优化算法的容忍度
    'max_iter': [1000, 5000, 10000]                 # 最大迭代次数
}
# 配置网格搜索（5折交叉验证）
grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=0
)

# 执行网格搜索（只在训练集上进行）
print("开始网格搜索...")
grid_search.fit(x_train, y_train)

# 输出最佳参数组合
print("\n最佳参数：")
print(grid_search.best_params_)

