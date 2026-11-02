"""
第一种尝试，增加流量数据，建立两个网络
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 平均速度
data_v = np.load(r'D:\01\traffic\pidl\PIDL数据+代码\新建文件夹\data\mat_v_i80.npz')['data']
# 平均密度
data_k = np.load(r'D:\01\traffic\pidl\PIDL数据+代码\新建文件夹\data\mat_k_i80.npz')['data']
# 平均流量
data_q = np.load(r'D:\01\traffic\pidl\PIDL数据+代码\新建文件夹\data\mat_q_i80.npz')['data']

# 清洗
data_v = pd.DataFrame(data_v)
data_v = data_v.fillna(0)
data_v = np.array(data_v)

data_k = pd.DataFrame(data_k)
data_k = data_k.fillna(0)
data_k = np.array(data_k)

data_q = pd.DataFrame(data_q)
data_q = data_q.fillna(0)
data_q = np.array(data_q)

# 相当于固定检测器位置，间隔5设置一个，相当于已知20%的数据，这个数字要设置成奇数（奇数这存疑，目前感觉是只要被150整除）
o = 5
observation = [i for i in range(0, 150, o)]
co = [i for i in range(199)]
for i in observation[:-1]:
    co.remove(i)

# 准备数据集
# observation = [0,3,5,8,10,12,15,17,19]
observation = [i for i in range(0, 150, o)]
# 抽样
v_train = np.zeros(data_v.shape)
k_train = np.zeros(data_k.shape)
q_train = np.zeros(data_q.shape)
for i in observation:
    v_train[i] = data_v[i]
    k_train[i] = data_k[i]
    q_train[i] = data_q[i]
v_train = v_train.reshape((1, 150, 300))
k_train = k_train.reshape((1, 150, 300))
q_train = q_train.reshape((1, 150, 300))
# 真实值
v_compare = data_v.reshape((1, 150, 300))
k_compare = data_k.reshape((1, 150, 300))
q_compare = data_q.reshape((1, 150, 300))

# 将numpy数据转化为torch的张量
inputs = torch.from_numpy(v_train).type(torch.FloatTensor)
targets = torch.from_numpy(v_compare).type(torch.FloatTensor)
inputs_1 = torch.from_numpy(q_train).type(torch.FloatTensor)
targets_1 = torch.from_numpy(q_compare).type(torch.FloatTensor)

D_in = 150*300
H = 50
H1 = 40
D_out = D_in


# 网络，含两层LSTM
class net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, H)
        self.linear8 = torch.nn.Linear(H, H)
        self.linear9 = torch.nn.Linear(H, H)
        self.lstm1 = torch.nn.LSTM(150, 256, 1, bias=True)
        self.lstm2 = torch.nn.LSTM(256, 150, 1, bias=True)
        self.linear10 = torch.nn.Linear(H, D_out)
        self.lrelu = torch.nn.ELU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.permute(0, 2, 1)  # 对输入张量x的维度进行重新排列
        x = self.lstm1(x)[0]
        x = torch.tanh(x)
        x = self.lstm2(x)[0]
        x = torch.sigmoid(x)
        x = x.view(1, -1)
        x = self.linear1(x)
#         x = torch.tanh(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        x = self.linear6(x)
        x = torch.tanh(x)
        x = self.linear7(x)
        x = torch.tanh(x)
        x = self.linear8(x)
        x = torch.tanh(x)
        x = self.linear9(x)
        x = torch.relu(x)
#         x = self.lstm(x)[0]
        x = self.linear10(x)
#         x = torch.relu(x)
        x = x.view(150, 300)
#         x = self.lrelu(x)
        return x

    def freeze_weights(self):
        self.freeze = True
        for p in self.parameters():
            p.requires_grad_(False)

    def un_freeze_weights(self):
        self.un_freeze = True
        for p in self.parameters():
            p.requires_grad_(True)


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    # mape=(abs(y_predict -y_test)/ y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    mape = []
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i, j] != 0:
                mape.append(abs(y_predict[i, j]-y_test[i, j])/y_test[i, j])
    return mae, rmse, np.mean(mape), r_2   # mse


# 损失: 守恒方程 P40,(3.18)
class My_loss(nn.Module):
    def __init__(self, dt=5.0, dx=10.0):
        super().__init__()
        self.dt = dt
        self.dx = dx

    def forward(self, x, x1):  # x为v_train，x1为q_train
        eps = 1e-4
        k = x1/(x + eps)
        q = x1
        dkdt = torch.gradient(k, dim=1)[0] * self.dt  # 考虑单位和采样尺度
        dqdx = torch.gradient(q, dim=0)[0] * self.dx
        dx = (dkdt+dqdx).type(torch.FloatTensor)
        e = torch.mean(torch.pow(dx, 2))
        return e

# 损失: Greenshields
class My_loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vf = 47
        self.kj = 0.19

    def forward(self, x, x1):  # x为v_train，x1为q_train
        k = self.kj*(1-x/self.vf)
        q = k*x
        q_c = x1
        e = torch.mean(torch.pow(q_c-q, 2))
        return e

# 损失: Greenberg
class My_loss3(nn.Module):
    def __init__(self):
        super().__init__()
        self.vf = 25.5
        self.kj = 0.23

    def forward(self, x, y):  # x为v_train，y为q_train
        k = self.kj*(torch.exp(-x/self.vf))
        q = k*x
        q_c = y
        e = torch.mean(torch.pow(q_c-q, 2))
        return e

# 损失
class My_loss4(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation =[i for i in range(0, 150, o)]
        self.co = [i for i in range(150)]
        for i in self.observation[:-1]:
            self.co.remove(i)

    def forward(self, x, y):
        # x = x[0]
        o_p = x[self.observation]
        # print(y.shape)
        o_t = y[0][self.observation]
        e1 = torch.mean(torch.pow((o_p - o_t), 2))

        return e1


# 训练
num_epochs = 1000
learning_rate = 0.005  # 0.005
l1 = 0.8
l2 = 0.9

# 权重系数
lamb_v = 1.0
lamb_q = 10.0  # 10.0
# lamb_pde = 1.0
# lamb_fd = 1.0

# 第三步： 实例化模型，构建两个网络模型
model_v = net(D_in, H1, D_out)
model_q = net(D_in, H1, D_out)

# 与模型相关的配置、损失函数、优化方式
# 使用自定义函数，等价于criterion = nn.MSELoss()
criterion = My_loss()
criterion_1 = My_loss4()
criterion_2 = My_loss2()

# 定义迭代优化算法， 使用的是随机梯度下降算法
optimizer = torch.optim.Adam(model_v.parameters(), lr=learning_rate, betas=(l1, l2), weight_decay=0.003)
optimizer_1 = torch.optim.Adam(model_q.parameters(), lr=learning_rate, betas=(l1, l2), weight_decay=0.003)
loss_history = []
last_loss = 1000

# 第四步：训练模型，迭代训练
for epoch in range(num_epochs):

    #  前向传播计算网络结构的输出结果
    outputs = model_v(inputs)
    outputs_1 = model_q(inputs_1)

    # 计算损失函数
    loss_v = criterion_1(outputs, targets)  # v
    loss_q = criterion_1(outputs_1, targets_1)  # q

    # 计算残差
    loss_pde = criterion(outputs, outputs_1)  # 守恒方程
    loss_fd = criterion_2(outputs, outputs_1)  # 基本图

    if epoch < 300:
        lamb_pde = 0.0
        lamb_fd = 0.0

    elif epoch < 600:
        lamb_pde = 0.1  # 0.1
        lamb_fd = 0.0  # 0.0

    else:
        lamb_pde = 1.0  # 1.0
        lamb_fd = 0.0  # 0.0

    all_loss = lamb_v*loss_v+lamb_q*loss_q+lamb_pde*loss_pde+lamb_fd*loss_fd
    # 超参数设置  k: 10 2000 1 1 / 10 5000 1 1 // q 100 20 1 5

    # 反向传播更新参数，三步策略，归零梯度——>反向传播——>更新参数
    optimizer.zero_grad()
    optimizer_1.zero_grad()
    all_loss.backward()

    optimizer.step()
    optimizer_1.step()

    # 打印训练信息和保存loss
    loss_history.append(all_loss.item())
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, all_loss.item()))

        last_loss = all_loss.item()
        predicted = model_q(torch.from_numpy(q_train).type(torch.FloatTensor)).detach().numpy()  # 模型输出结果

        mae, rmse, mape, r_2 = evaluation(data_q, predicted)
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('MAPE:', mape)
        print('R2:', r_2)

predicted = model_v(torch.from_numpy(v_train).type(torch.FloatTensor)).detach().numpy()  # 模型输出结果
# predicted_k = model_k(torch.from_numpy(k_train).type(torch.FloatTensor)).detach().numpy()
predicted_q = model_q(torch.from_numpy(q_train).type(torch.FloatTensor)).detach().numpy()
# print(predicted)

# 第五步：结果展示
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 2)
plt.imshow(predicted_q)
plt.title('flow prediction')

plt.subplot(1, 2, 1)
plt.imshow(data_q)
plt.title('flow reality')
# plt.plot(x_train, predicted, label='Fitted line')             # 拟合之后的直线

# plt.subplot(1, 2, 1)
# plt.imshow(data_v)
# plt.title('velocity reality')
#
# plt.subplot(1, 2, 2)
# plt.imshow(predicted)
# plt.title('velocity prediction')

plt.tight_layout()

plt.show()
