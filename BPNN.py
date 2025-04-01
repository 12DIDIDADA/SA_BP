# -*- coding: utf-8 -*-
import math

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch.nn.functional as F
# torch.nn.functional.leaky_relu
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv',encoding="gbk")
data = data.rename(columns={'地层':'Stratum','土地利用':'landuse', '土壤类型':'soiltype'})
x1 = data[['F1','F2','F3','F4','Stratum','landuse','soiltype']]

y1 = data['As']

x_train,x_temp, y_train, y_temp = train_test_split(x1, y1, test_size=0.3, random_state=88)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1/2, random_state=88)# 数据类型转换
x1 = x1.astype(np.float32)

y1 = y1.astype(np.float64)

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
x_val = transfer.transform(x_val)
x1 = transfer.transform(x1)

# 构建 PyTorch 数据集对象
train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values.reshape(-1, 1)))
test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.values.reshape(-1, 1)))

train_dataset, test_dataset, input_dim, class_num = (
    train_dataset, test_dataset, x_train.shape[1], len(np.unique(y_train)))

class pre(nn.Module):

    def __init__(self, input_num):
        super(pre, self).__init__()
        self.linear1 = nn.Linear(input_num, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=128)
        self.linear4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.prelu1 = nn.ELU()
        self.prelu2 = nn.ELU()


    def forward(self, x):
        x = self.prelu1(self.linear1(x))
        x = self.prelu2(self.linear2(x))
        output = self.linear4(x)
        return output



from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(88)
model = pre(input_dim).to('cuda')
criterion = nn.HuberLoss().to('cuda') #损失函数

optimizer = optim.RMSprop(model.parameters(), lr=0.01,weight_decay=0.05) #优化方法

num_epochs = 100
true_values = []
predicted_values = []  # 存储正确与预测的

# 训练模型
for epoch_idx in range(num_epochs):
    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    start = time.time()
    total_loss = 0
    total_num = 0
    total_correct = 0

    for x, y in data_loader:
        x = x.float().to('cuda')
        y = y.float().to('cuda')
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += len(y)
        total_loss += loss.item() * len(y)

        # 将真实值和预测值添加到列表中
        true_values.extend(y.cpu().tolist())
        predicted_values.extend(output.cpu().tolist())

    r2 = r2_score(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)

    print('epochs: %4s loss: %.6f R2: %.3f MSE: %.3f MAE: %.3f time:%.3fs' %
          (epoch_idx + 1,
           total_loss / total_num,
           r2,
           mse,
           mae,
           time.time() - start))

    # 学习率调度器更新
    val_loss = total_loss / total_num
    scheduler.step(val_loss)

    # 验证过程
model.eval()
with torch.no_grad():
    x_test_tensor = torch.tensor(x_test).float().to('cuda')  # 转换为 PyTorch 张量
    y_test_tensor = torch.tensor(y_test.values).float().view(-1, 1).to('cuda')

    # 前向传播计算输出
    outputs = model(x_test_tensor)

    # 计算损失和评估指标
    test_loss = criterion(outputs, y_test_tensor)
    test_mse = mean_squared_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())  # MSE
    test_mae = mean_absolute_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())  # MAE
    test_r2 = r2_score(y_test_tensor.cpu().numpy(), outputs.cpu().numpy())  # R²

    # 输出评估结果
    print(f"Test Loss: {test_loss.item():.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")





torch.save(model.state_dict(), r'/root/model/As_BPNN_pre.pth')
# 测试集
x_array = x_val.astype(np.float32) # 转换 DataFrame 为 NumPy 数组
x_array_all = x1.astype(np.float32) # 转换 DataFrame 为 NumPy 数组
x_array_test = x_test.astype(np.float32) # 转换 DataFrame 为 NumPy 数组

transfer = StandardScaler()
x_array = transfer.fit_transform(x_array)

# 将 NumPy 数组转换为 torch 张量，并转换为浮点数类型
x_tensor = torch.tensor(x_array, dtype=torch.float32)
x_tensor_all = torch.tensor(x_array_all, dtype=torch.float32)
x_tensor_test = torch.tensor(x_array_test, dtype=torch.float32)
# 加载模型参数

model_state_dict = torch.load(r'/root/model/As_BPNN_pre.pth')


# 设置输入特征维度
input_dim = x_array.shape[1]
# 创建模型
model = pre(input_dim)
model.load_state_dict(model_state_dict)


# 进行预测
model.eval()
with torch.no_grad():  # 关闭梯度计算，用于推断
    output = model(x_tensor_test)
df_test = pd.DataFrame()
df_test['As'] = y_test
df_test['预测'] = output.numpy()
r_all = r2_score(df_test['As'],df_test['预测']).__round__(3)
mae_all = mean_absolute_error(df_test['As'],df_test['预测']).__round__(3)
mse_all = mean_squared_error(df_test['As'],df_test['预测']).__round__(4)
print(f"验证集r2:{r_all}\nmae:{mae_all} \nmse:{mse_all}")
df_test.to_csv(r'/root/BPNN/As_test.csv',index=False)


# 进行预测
model.eval()
with torch.no_grad():  # 关闭梯度计算，用于推断
    output = model(x_tensor)
df_val = pd.DataFrame()
df_val['As'] = y_val
df_val['预测'] = output.numpy()
r_val = r2_score(df_val['As'],df_val['预测']).__round__(3)
mae_val = mean_absolute_error(df_val['As'],df_val['预测']).__round__(3)
mse_val = mean_squared_error(df_val['As'],df_val['预测']).__round__(4)
print(f"测试集r2:{r_val}\nmae:{mae_val} \nmse:{mse_val}")

df_val.to_csv(r'/root/BPNN/As_val.csv',index=False)


# 进行预测
model.eval()
with torch.no_grad():  # 关闭梯度计算，用于推断
    output = model(x_tensor_all)
df_all = pd.DataFrame()
df_all['As'] = y1
df_all['预测'] = output.numpy()
r_all = r2_score(df_all['As'],df_all['预测']).__round__(3)
mae_all = mean_absolute_error(df_all['As'],df_all['预测']).__round__(3)
mse_all = mean_squared_error(df_all['As'],df_all['预测']).__round__(4)
print(f"所有的r2:{r_all}\nmae:{mae_all} \nmse:{mse_all}")

df_all.to_csv(r'/root/BPNN/As_all.csv',index=False)