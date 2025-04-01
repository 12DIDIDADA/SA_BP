#无正则化、随机种子88
import csv
import json
import random
import time
from scipy.optimize import dual_annealing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import warnings
warnings.filterwarnings("ignore")


seed = 88
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data = pd.read_csv(r'/root/data/ms_pca_fanzhuan.csv',encoding=  'gbk')
# data =  pd.read_csv(r"D:\arcpy_io\arcmap_data\PCA\ms_pca_fanzhuan.csv",encoding=  'gbk')
x1 = data[['F1','F2','F3','F4','地层','土地利用','土壤类型']]
y1 = data['As']
x_train, x_temp, y_train, y_temp = train_test_split(x1, y1, test_size=0.3, random_state=88)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=1 / 2, random_state=88)


x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
x_val = x_val.astype(np.float32)
y_val = y_val.astype(np.float32)
x1 = x1.astype(np.float32)
y1 = y1.astype(np.float32)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x1 = scaler.transform(x1)

train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values.reshape(-1, 1)))
test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.values.reshape(-1, 1)))
val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.values.reshape(-1, 1)))
all_dataset = TensorDataset(torch.from_numpy(x1), torch.from_numpy(y1.values.reshape(-1, 1)))

input_dim = x1.shape[1]
#自定义的差值损失函数
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

# 定义模型
class bpnn(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units_range, dropout_rate, activation_func):
        super(bpnn, self).__init__()
        self.hidden_layers = hidden_layers
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        layers = []
        prev_units = input_dim
        self.hidden_units_list = [random.randint(hidden_units_range[0], hidden_units_range[1]) for _ in range(hidden_layers)]
        self.layer_details = []
        #遍历神经元的表
        for hidden_units in self.hidden_units_list:
            layers.append(nn.Linear(prev_units, hidden_units))
            self.layer_details.append((prev_units, hidden_units))
            prev_units = hidden_units
            layers.append(self.get_activation_function())
            layers.append(nn.Dropout(p=self.dropout_rate))

        layers.append(nn.Linear(prev_units, 1))
        self.layer_details.append((prev_units, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def get_activation_function(self):
        if self.activation_func == 0:
            return nn.ReLU()
        elif self.activation_func == 1:
            return nn.Tanh()
        elif self.activation_func == 2:
            return nn.Sigmoid()
        elif self.activation_func == 3:
            return nn.ELU()
        elif self.activation_func == 4:
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation_func == 5:
            return nn.GELU()
        elif self.activation_func == 6:
            class Swish(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x):
                    return x * torch.sigmoid(x)
            return Swish()

best_params_dict = {            '隐藏层数量': 1,
                                '每层神经元数量': 1,
                                '丢弃率': 1,
                                '激活函数索引': 1,
                                '学习率': 1,
                                '批次大小': 1,
                                '迭代次数': 1,
                                '损失函数索引': 1,
                                '优化器索引': 1,
                                '评估指标': 1}  # 确保全局变量初始化
best_dict = {            '隐藏层数量': 1,
                         '每层神经元数量': 1,
                         '丢弃率': 1,
                         '激活函数索引': 1,
                         '学习率': 1,
                         '批次大小': 1,
                         '迭代次数': 1,
                         '损失函数索引': 1,
                         '优化器索引': 1,
                         '评估指标': 1}
best_eval_score = 0  # 用来保存最优的评估值
best_train_score = 0  # 用来保存最优的评估值
best_val_score = 0  # 用来保存最优的验证集损失
start_time = time.time()
num_a = 0
time_use = 0
def fitness_function(params):
    global best_params_dict
    global best_dict
    global num_a
    global time_use
    # 检查无效参数
    if np.isnan(params).any() or np.isinf(params).any():
        return float('inf')
    global best_eval_score  # 使用全局变量
    global best_train_score
    global train_r2
    global last_r2
    hidden_units_range = [50, 400]
    hidden_layers = int(round(params[0]))
    if time.time() - start_time > 3600 * 24 * 3:
        return float('inf')
    if hidden_layers < 1 or hidden_layers > 3:
        return float('inf')

    dropout_rate = round(params[2], 2)
    activation_func_index = int(round(params[3]))
    learning_rate = round(params[4], 2)
    batch_size = int(round(params[5]))
    num_epochs = int(round(params[6]))
    loss_func_index = int(round(params[7]))
    optimizer_index = int(round(params[8]))

    # 检查 hidden_units_range 是否有效
    if hidden_units_range[0] >= hidden_units_range[1]:
        print("无效的神经元数量范围，跳过当前参数组合。")
        return float('inf')

    hidden_units_list = [random.randint(hidden_units_range[0], hidden_units_range[1]) for _ in range(hidden_layers)]
    print(f"生成的每层神经元数量: {hidden_units_list}")

    # 选择损失函数
    criterion = {
        0: nn.MSELoss().to('cuda'),
        1: nn.L1Loss().to('cuda'),
        2: nn.HuberLoss().to('cuda'),
        3: LogCoshLoss().to('cuda'),  # 新增LogCosh损失
        4: nn.SmoothL1Loss().to('cuda')  # 新增SmoothL1Loss
    }[loss_func_index]

    # 选择优化器
    optimizer_name = {
        0: 'Adam',
        1: 'SGD',
        2: 'RMSprop',
        3: 'Adagrad',  # 新增Adagrad
        4: 'NAdam'     # 新增NAdam
    }[optimizer_index]

    # 检查激活函数索引
    activation_functions = ['ReLU', 'Tanh', 'Sigmoid', 'ELU', 'LeakyReLU', 'GELU', 'Swish']  # 扩展列表
    # 获取激活函数显示名称
    activation_func_display = activation_functions[activation_func_index]
    # 检查激活函数索引的有效性
    if activation_func_index < 0 or activation_func_index >= len(activation_functions):
        print("无效的激活函数索引，跳过当前参数组合。")
        return float('inf')
    # 检查损失函数索引
    if loss_func_index < 0 or loss_func_index > 4:
        print("无效的损失函数索引，跳过当前参数组合。")
        return float('inf')

    # 检查优化器索引
    if optimizer_index < 0 or optimizer_index > 4:
        print("无效的优化器索引，跳过当前参数组合。")
        return float('inf')



    # 打印当前参数
    print(f"当前参数: 隐藏层数量={hidden_layers}, 丢弃率={dropout_rate}, 激活函数={activation_functions[activation_func_index]}, "
          f"学习率={learning_rate}, 批次大小={batch_size}, 迭代次数={num_epochs}, "
          f"损失函数={['MSELoss', 'L1Loss', 'HuberLoss','LogCoshLoss','SmoothL1Loss'][loss_func_index]}, 优化器={optimizer_name}, "
          f"神经元数量={hidden_units_list}")

    model = bpnn(input_dim, hidden_layers, hidden_units_range, dropout_rate, activation_func_index).to('cuda')
    optimizer = {
        0: optim.Adam(model.parameters(), lr=learning_rate),
        1: optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        2: optim.RMSprop(model.parameters(), lr=learning_rate),
        3: optim.Adagrad(model.parameters(), lr=learning_rate),  # 新增Adagrad
        4: optim.NAdam(model.parameters(), lr=learning_rate)      # 新增NAdam
    }[optimizer_index]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    negative_r2_count = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    train_r2 = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_true = []
        all_pred = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.float().to('cuda'), y_batch.float().to('cuda')
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

            all_true.extend(y_batch.cpu().tolist())
            all_pred.extend(outputs.cpu().tolist())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        if np.isnan(all_true).any() or np.isinf(all_true).any() or np.isnan(all_pred).any() or np.isinf(all_pred).any():
            print("检测到无效值，返回无穷大。")
            return float('inf')

        r2 = r2_score(all_true, all_pred)
        mse = mean_squared_error(all_true, all_pred)
        mae = mean_absolute_error(all_true, all_pred)
        train_r2.append(r2)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, 损失: {total_loss / len(train_loader.dataset):.4f}, R2: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}')
        if r2 < 0:
            negative_r2_count += 1
        else:
            negative_r2_count = 0

        if negative_r2_count >= 3:
            print(f"连续{negative_r2_count}次R²小于0，取消当前参数组合。")
            return float('inf')

    scheduler.step(total_loss / len(train_loader.dataset))
    last_r2 = train_r2[-1]
    print('开始验证集评估')
    # 验证集评估
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 使用 DataLoader 加载测试数据
        test_true = []
        test_pred = []

        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.float().to('cuda'), y_batch.float().to('cuda')
            outputs = model(x_batch)
            test_true.extend(y_batch.cpu().tolist())
            test_pred.extend(outputs.cpu().tolist())

        test_true = np.array(test_true).ravel()
        test_pred = np.array(test_pred).ravel()

        test_mse = mean_squared_error(test_true, test_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        test_r2 = r2_score(test_true, test_pred)

    print(f"验证集评估结果: MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    # 将结果保存为 CSV 格式
    results_df_test = pd.DataFrame({
        'True Values': test_true,
        'Predicted Values': test_pred
    })

    print('开始测试集评估')
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 使用 DataLoader 加载测试数据
        val_true = []
        val_pred = []

        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.float().to('cuda'), y_batch.float().to('cuda')
            outputs = model(x_batch)
            val_true.extend(y_batch.cpu().tolist())
            val_pred.extend(outputs.cpu().tolist())

        val_true = np.array(val_true).ravel()
        val_pred = np.array(val_pred).ravel()

        val_mse = mean_squared_error(val_true, val_pred)
        val_mae = mean_absolute_error(val_true, val_pred)
        val_r2 = r2_score(val_true, val_pred)

    print(f"测试集评估结果: MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    # 将结果保存为 CSV 格式
    results_df_val = pd.DataFrame({
        'True Values': val_true,
        'Predicted Values': val_pred
    })

    with torch.no_grad():
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)  # 使用 DataLoader 加载测试数据
        all_true = []
        all_pred = []

        for x_batch, y_batch in all_loader:
            x_batch, y_batch = x_batch.float().to('cuda'), y_batch.float().to('cuda')
            outputs = model(x_batch)
            all_true.extend(y_batch.cpu().tolist())
            all_pred.extend(outputs.cpu().tolist())

        all_true = np.array(all_true).ravel()#展开成一维数组好放入后面
        all_pred = np.array(all_pred).ravel()

        all_mse = mean_squared_error(all_true, all_pred)
        all_mae = mean_absolute_error(all_true, all_pred)
        all_r2 = r2_score(all_true, all_pred)

    print(f"全部评估结果: MSE: {all_mse:.4f}, MAE: {all_mae:.4f}, R²: {all_r2:.4f}")
    # 将结果保存为 CSV 格式
    results_df_all = pd.DataFrame({
        'True Values': all_true,
        'Predicted Values': all_pred
    })

    results_df_all1 = results_df_all
    results_df_val1 = results_df_val
    results_df_test1 = results_df_test
    current_eval_score = test_r2
    current_train_score = r2#train_r2-1
    current_val_score = val_r2
    current_all_score = all_r2
    # current_val_score = val_mae
    # 如果当前评估结果比之前的最优值更好，更新最优参数
    if current_eval_score > best_eval_score :
        num_a += 1
        time_use = time.time() - start_time
        best_eval_score = current_eval_score
        best_train_score = current_train_score
        val_score1 = current_val_score
        all_score1 = current_all_score
        # 打开文件（追加模式，如果文件不存在会自动创建）
        with open('/root/SA_BPNN/num_a/提交版.csv', 'a', newline='') as file:
            # with open(r'D:\python\arcpy_learn\workplace\arcpy_test1\4090\DATA\111.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            # 检查文件是否为空（是否需要写入文件头）
            if file.tell() == 0:
                writer.writerow(['num_a', 'train', 'test','val','all','time_use'])  # 写入列名

            # 写入数据
            writer.writerow([num_a, best_train_score, best_eval_score,val_score1,all_score1,time_use])
        # best_all_score = current_all_score
        results_df_all1.to_csv('/root/return/提交版_all.csv', index=False,float_format='%.4f')
        results_df_val1.to_csv('/root/return/提交版_val.csv', index=False,float_format='%.4f')
        results_df_test1.to_csv('/root/return/提交版_test.csv', index=False,float_format='%.4f')

        torch.save(model.state_dict(), r'/root/model/SA_BPNN/提交版.pth')
        # 更新最优参数字典
        best_params_dict = {
            '隐藏层数量': hidden_layers,
            '每层神经元数量': hidden_units_list,
            '丢弃率': dropout_rate,
            '激活函数索引': activation_func_index,
            '学习率': learning_rate,
            '批次大小': batch_size,
            '迭代次数': num_epochs,
            '损失函数索引': loss_func_index,
            '优化器索引': optimizer_index,
            '评估指标': current_eval_score  # 这里保存的是 MAE
        }
        # 改进后的best_dict生成方式，使用正确的键名来获取值
        best_dict = {
            '隐藏层数量': int(round(best_params_dict['隐藏层数量'])),
            '每层神经元数量': best_params_dict['每层神经元数量'],
            '丢弃率': round(best_params_dict['丢弃率'], 2),
            '激活函数索引': int(round(best_params_dict['激活函数索引'])),
            '学习率': round(best_params_dict['学习率'], 2),
            '批次大小': int(round(best_params_dict['批次大小'])),
            '迭代次数': int(round(best_params_dict['迭代次数'])),
            '激活函数名称': ['ReLU', 'Tanh', 'Sigmoid', 'ELU', 'LeakyReLU', 'GELU', 'Swish'][int(round(best_params_dict['激活函数索引']))],
            '损失函数名称': ['MSELoss', 'L1Loss', 'HuberLoss', 'LogCosh', 'SmoothL1'][int(round(best_params_dict['损失函数索引']))],
            '优化器名称': ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'NAdam'][int(round(best_params_dict['优化器索引']))]
        }

        # 直接打印更新后的最优参数字典
        print("更新后的最优参数：")
        print(best_dict)  # 直接打印字典
    print('结束' )  # 直接打印字典
    print(best_dict )  # 直接打印字典
    with open(r'/root/data2/提交版.json', 'w', encoding='utf-8') as f:
        # with open(r"D:\python\arcpy_learn\workplace\arcpy_test1\mashan\data\As_SA+BPNN.json", 'w', encoding='utf-8') as f:
        json.dump(best_dict, f, ensure_ascii=False, indent=4)
    # 返回目标值用于优化
    return current_eval_score


# 模拟退火参数
bounds = [
    (1, 3),  # 隐藏层数量
    (50, 400),  # 每层神经元数量
    (0.1, 0.5),  # 丢弃率
    (0, 6),  # 激活函数选择
    (0.01, 0.2),  # 学习率
    (1, 4),  # 批次大小
    (50, 400),  # 迭代次数
    (0, 4),  # 损失函数选择
    (0, 4)   # 优化器选择
]
current_temp = 1000  # 初始化当前温度

def update_temp(x, e, context):
    global current_temp
    current_temp *= 0.9
    print(f"当前温度: {current_temp:.2f}")  # 打印当前温度
    if current_temp < 1:  # 设定一个最小温度阈值
        print("达到最小温度，停止优化。")
        return True  # 返回 True 来指示停止优化
    return False  # 继续优化


options = {
    'initial_temp': 1000,
    'maxiter': 1000,
    'method': 'BFGS'  # 使用的局部搜索算法
}

result = dual_annealing(
    fitness_function,
    bounds,
    seed=88,
    maxiter=options['maxiter'],
    initial_temp=options['initial_temp'],

    callback=update_temp
)


