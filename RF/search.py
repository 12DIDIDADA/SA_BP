import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap

# 读取数据
data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv', encoding='gbk')
x = data[['F1', 'F2', 'F3', 'F4', '地层', '土地利用', '土壤类型']]
y = data['As']

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 数据集划分
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=88)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1/2, random_state=88)

# 定义参数网格
param_grid = {
    'n_estimators': [ 100, 200, 300],
    'max_depth': [10,20,30],
    'min_samples_split': [1, 3],
    'min_samples_leaf': [1, 3 ],
    'criterion': ['squared_error', 'absolute_error'],

}

# 创建随机森林模型
rf_model = RandomForestRegressor(random_state=88)

# 配置网格搜索
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=2
)

# 执行网格搜索
grid_search.fit(x_train, y_train)

# 输出最佳参数
print("最佳参数：")
print(grid_search.best_params_)
