"""
第二种尝试，ARZ模型
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

# 平均速度
data = np.load(r'D:\01\traffic\pidl\PIDL数据+代码\新建文件夹\data\mat_v_i80.npz')['data']

data = pd.DataFrame(data)
data = data.fillna(0)
data = np.array(data)

# 相当于固定检测器位置，间隔5设置一个，相当于已知30%的数据，这个数字要设置成奇数（奇数这存疑，目前感觉是只要被150整除）
o = 3
observation = [i for i in range(0, 150, o)]
co = [i for i in range(199)]
for i in observation[:-1]:
    co.remove(i)

# 准备数据集
# observation = [0,3,5,8,10,12,15,17,19]
observation = [i for i in range(0, 150, o)]
x_train = np.zeros(data.shape)
for i in observation:
    x_train[i] = data[i]
x_train = x_train.reshape((1, 150, 300))
y_train = data.reshape((1, 150, 300))

# 将numpy数据转化为torch的张量
inputs = torch.from_numpy(x_train).type(torch.FloatTensor)
targets = torch.from_numpy(y_train).type(torch.FloatTensor)
# print('input_size:', inputs.shape)
# print('target_size:', targets.shape)

D_in = 150*300  # 输入特征的维度
H = 50  # 隐藏层的维度
H1 = 40
D_out = D_in  # 输出特征的维度
# vf = 22.22

# 第一步：准备网络
# 两层LSTM
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
        self.lstm1 = torch.nn.LSTM(150, 256, 1, bias=True)  # 输入特征维度为150，隐藏状态维度为256，层数为1，使用偏置项
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
        x = x.view(1, -1)  # 输出展平为一个二维张量，形状为(1, -1)
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


# 全连接，无LSTM
class net_origin(torch.nn.Module):
    def __init__(self, D_in, H1, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(net_origin, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H1)
        self.linear3 = torch.nn.Linear(H1, H1)
        self.linear4 = torch.nn.Linear(H1, H1)
        self.linear5 = torch.nn.Linear(H1, H1)
        self.linear6 = torch.nn.Linear(H1, H1)
        self.linear7 = torch.nn.Linear(H1, H1)
        self.linear8 = torch.nn.Linear(H1, H1)
        self.linear9 = torch.nn.Linear(H1, H1)
#         self.lstm1 = torch.nn.LSTM(200,256, 1, bias = True)
#         self.lstm2 = torch.nn.LSTM(256,200, 1, bias = True)
        self.linear10 = torch.nn.Linear(H1, D_out)
#         self.lrelu = torch.nn.ELU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
#         x = x.permute(0,2,1)
#         x = self.lstm1(x)[0]
#         x = torch.tanh(x)
#         x = self.lstm2(x)[0]
#         x = torch.sigmoid(x)
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


# 第二步：构造损失函数
# ARZ+Greenshields
class My_loss7(nn.Module):
    def __init__(self):
        super().__init__()
        self.vf = 47
        self.kj = 0.19
        self.observation =[i for i in range(0, 150, o)]
        self.co = [i for i in range(150)]
        for i in self.observation[:-1]:
            self.co.remove(i)

    def forward(self, x, y):
        o_p = x[self.observation]
        o_t = y[0][self.observation]
        e1 = torch.mean(torch.pow((o_p - o_t), 2))

        k = self.kj*(1-x/self.vf)
        q = k*x
        dkdt = torch.gradient(k, dim=1)[0]*5  # 考虑单位和采样尺度
        dqdx = torch.gradient(q, dim=0)[0]*10
        dx = (dkdt + dqdx).type(torch.FloatTensor)
        e2 = torch.mean(torch.pow(dx, 2))
        # P2 (8) \gama = 2
        dvdt = torch.gradient(x, dim=1)[0] * 5
        dvdx = torch.gradient(x, dim=0)[0] * 10
        h = x - 2*pow(k, 2)/pow(self.kj, 2)*self.vf
        du = dvdt + dvdx*(h*k+x)
        e3 = torch.mean(torch.pow(du, 2))
        return e1+100*e2+0.001*e3


# ARZ+Greenberg
class My_loss8(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vf = 43.24
        self.vf = 25.5
        self.kj = 0.23
        self.observation =[i for i in range(0, 150, o)]
        self.co = [i for i in range(150)]
        for i in self.observation[:-1]:
            self.co.remove(i)

    def forward(self, x, y):

        o_p = x[self.observation]
        o_t = y[0][self.observation]
        e1 = torch.mean(torch.pow((o_p - o_t), 2))
        k = self.kj*(torch.exp(-x/self.vf))
        q = k*x
        dkdt = torch.gradient(k, dim=1)[0]*5
        dqdx = torch.gradient(q, dim=0)[0]*10
        dx = (dkdt+dqdx).type(torch.FloatTensor)
        e2 = torch.mean(torch.pow(dx, 2))
        # P2 (8) \gama = 2
        dvdt = torch.gradient(x, dim=1)[0] * 5
        dvdx = torch.gradient(x, dim=0)[0] * 10
        h = x - 2 * pow(k, 2) / pow(self.kj, 2) * self.vf
        du = dvdt + dvdx * h
        e3 = torch.mean(torch.pow(du, 2))
        return e1+1000*e2+0.0001*e3


# 训练
num_epochs = 1000
learning_rate = 0.005
l1 = 0.8
l2 = 0.9

# 第三步： 实例化模型，构建一个一层的网络模型
model = net(D_in, H1, D_out)

# 与模型相关的配置、损失函数、优化方式
# 使用自定义函数，等价于criterion = nn.MSELoss()
criterion = My_loss7()

# 定义迭代优化算法， 使用的是随机梯度下降算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(l1, l2), weight_decay=0.003)
loss_history = []
last_loss = 1000

# 第四步：训练模型，迭代训练
for epoch in range(num_epochs):

    #  前向传播计算网络结构的输出结果
    outputs = model(inputs)

    # 计算损失函数
    loss = criterion(outputs, targets)

    # 反向传播更新参数，三步策略，归零梯度——>反向传播——>更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息和保存loss
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        last_loss = loss.item()
        predicted = model(torch.from_numpy(x_train).type(torch.FloatTensor)).detach().numpy()  # 模型输出结果
        mae, rmse, mape, r_2 = evaluation(data, predicted)
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('MAPE:', mape)
        print('R2:', r_2)

predicted = model(torch.from_numpy(x_train).type(torch.FloatTensor)).detach().numpy()  # 模型输出结果

# 第五步：结果展示
plt.figure(figsize=(10, 4))
prediction = predicted
plt.subplot(1, 2, 2)
plt.imshow(prediction)
plt.title('prediction')

plt.subplot(1, 2, 1)
plt.imshow(data)
plt.title('reality')
# plt.plot(x_train, predicted, label='Fitted line')             # 拟合之后的直线

plt.show()
