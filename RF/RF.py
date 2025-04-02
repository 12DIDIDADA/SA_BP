import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
# 读取数据
data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv',encoding=  'gbk')
x = data[['F1', 'F2', 'F3', 'F4', '地层', '土地利用', '土壤类型']]
y = data['As']

# 数据标准化
scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=88)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1/2, random_state=88)




# 初始化模型
rf_model = RandomForestRegressor(n_estimators=200,   #决策树数量
                                 criterion="absolute_error",#absolute_error
                                 min_samples_split=3, #非叶子节点最小
                                 min_samples_leaf=3,#叶子节点最小
                                 max_depth=10, #决策树最大深度
                                 random_state=88)

# 存储每一折的结果
r2_scores = []
mse_scores = []
mae_scores = []

rf_model.fit(x_train, y_train)

# 预测
y_pred = rf_model.predict(x_test)

# 计算评估指标
R2_score = r2_score(y_test, y_pred)
MSE_score = mean_squared_error(y_test, y_pred)
MAE_score = mean_absolute_error(y_test, y_pred)

# 存储每一折的结果
r2_scores.append(R2_score)
mse_scores.append(MSE_score)
mae_scores.append(MAE_score)


print("\n验test的平均结果：")
print(f"平均R²分数: {np.mean(r2_scores):.2f}")
print(f"平均MSE分数: {np.mean(mse_scores):.4f}")
print(f"平均MAE分数: {np.mean(mae_scores):.3f}")
data_test = pd.DataFrame()
data_test['y_test'] = y_test
data_test['y_test_pred'] = y_pred
data_test.to_csv(r'/root/RF/RF_tet.csv', index=False)
# 在测试集上评估模型
y_val_pred = rf_model.predict(x_val)
R2_val = r2_score(y_val, y_val_pred)
MSE_val = mean_squared_error(y_val, y_val_pred)
MAE_val = mean_absolute_error(y_val, y_val_pred)

print("\n在val上的评估结果：")
print(f"R²分数: {R2_val:.2f}, MSE分数: {MSE_val:.4f}, MAE分数: {MAE_val:.3f}")
data_val = pd.DataFrame()
data_val['y_val'] = y_val
data_val['y_val_pred'] = y_val_pred
data_val.to_csv(r'/root/RF/RF_val.csv', index=False)



