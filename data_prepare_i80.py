import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 总体数据
data = pd.read_csv(r'D:\01\traffic\pidl\PIDL数据+代码\新建文件夹\PIDL_data\NGSIM_Data.csv')
data['Location'].unique()

# 筛选道路
road = 'i-80'
data = data[data['Location'] == road]
plt.axis('equal')

# Local_X是车辆在当前车道上的横向位置
x = data['Local_X']
# Local_Y是车辆在当前车道上的纵向位置
y = data['Local_Y']
# plt.scatter(x, y)

# Global_Time是全局时间
data = data.sort_values('Global_Time')
a = data['Global_Time'].max()-data['Global_Time'].min()

# 时空划分
x_space = 10   # 单位英尺（0.3m）
t_space = 5000  # 频率为1000fps
loca = [i for i in range(100, 2400, x_space)]
Time = [i for i in range(data['Global_Time'].min(), data['Global_Time'].max()+t_space, t_space)]

mat = []  # 储存平均流量
mat_v = []  # 储存平均速度
for i in loca:  # 空间
    tem_local = data[data['Local_Y'] < i]
    tem_local_ = tem_local[tem_local['Local_Y'] > (i-x_space)]
    nat = []
    nat_v = []
    for j in Time:  # 时间
        tem_time = tem_local_[tem_local_['Global_Time'] > j]
        tem_time_ = tem_time[tem_time['Global_Time'] < (j+t_space)]
        # 算平均每帧的车辆数量，即平均密度
        num = len(tem_time_)/t_space*100
        # 算平均速度
        v_m = np.mean(tem_time_['v_Vel'])  # v_Vel是车辆当前的瞬时速度
        nat.append(num)
        nat_v.append(v_m)
    mat.append(nat)
    mat_v.append(nat_v)
mat = np.array(mat)
mat_v = np.array(mat_v)

# plt.subplot(1, 2, 1)
# plt.imshow(mat_v[:150, 750:1050])  # 适当截取
# plt.subplot(1, 2, 2)
# plt.scatter(mat_v.reshape(-1), mat.reshape(-1), s=1)
# plt.show()

mat_q = mat_v*mat

# 保存
np.savez('mat_q_i80.npz', data=mat_q[:150, 750:1050])
# np.savez('mat_v_i80.npz', data=mat_v[:150, 750:1050])
