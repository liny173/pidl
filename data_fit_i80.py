import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Greenberg模型
def golinbo(x, a, b):
    return b*np.exp(-x/a)


# Greenshields模型
def greenshields(x, a, b):
    return a*(1-x/b)


# Van_Aerde模型
def Van_Aerde(x, vf, vm, kj, qm):
    c1 = vf*(2*vm-vf)/(kj*vm*vm)
    c2 = vf*(vf-vm)*(vf-vm)/(kj*vm*vm)
    c3 = 1/qm-vf/(kj*vm*vm)
    return 1/(c1+c2/(vf-x)+c3*x)


# 导入数据
mat = np.load(r'D:\01\traffic\pidl\PIDL\新建文件夹\data\mat_k_i80.npz')['data']
mat_v = np.load(r'D:\01\traffic\pidl\PIDL\新建文件夹\data\mat_v_i80.npz')['data']

# 转化和清洗
mat_v = pd.DataFrame(mat_v)
mat_v = mat_v.fillna(0)
mat_v = np.array(mat_v)[:, :-1]
mat = pd.DataFrame(mat)
mat = mat.fillna(0)
mat = np.array(mat)[:, :-1]*0.1
mat = mat.astype(np.float64)
mat_v = mat_v.astype(np.float64)

# 拟合
# Greenberg
popt, pcov = curve_fit(golinbo, mat_v.reshape(-1), mat.reshape(-1))

# Greenshields
initialParameters = np.array([1.0, 1.0])
popt_1, pcov_1 = curve_fit(greenshields, mat_v.reshape(-1), mat.reshape(-1),initialParameters)

# Van Aerde
initialParameters_1 = np.array([1.0, 1.0, 0.1, 1.0])
popt_2, pcov_2 = curve_fit(Van_Aerde, mat_v.reshape(-1), mat.reshape(-1),initialParameters_1, bounds=(0, [100, 100, 0.28,50]))

# 参数输出
# Greenberg的vf和kj
a = popt[0]
b = popt[1]
print(f'Greenberg p_j:{b}\nV_m:{a}')

# Greenshields的vf和kj
a_1 = popt_1[0]
b_1 = popt_1[1]
print(f'Greenshields p_j:{b_1}\nV_f:{a_1}')

# Van Aerde的vf、kj、Qm、vm
a_2 = popt_2[2]
b_2 = popt_2[0]
c = popt_2[3]
d = popt_2[1]
print(f'Van_Aerde p_j:{b_2}\nV_f:{a_2}\nq_m:{c}\nV_m:{d}')

# 绘图
# 横坐标
x = [i for i in range(50)]
x = np.array(x)
x = x.astype(np.float64)

plt.figure(figsize=(16, 4))

# 子图: Greenberg
plt.subplot(1, 3, 1)
plt.scatter(mat_v.reshape(-1), mat.reshape(-1), s=0.5)
plt.plot(x, golinbo(x, *popt), 'r')
plt.ylabel('p(veh/ft)')
plt.xlabel('V(ft/s)')
plt.title('Greenberg')

# 子图: Greenshields
plt.subplot(1, 3, 2)
plt.scatter(mat_v.reshape(-1), mat.reshape(-1), s=0.5)
plt.plot(x, greenshields(x, *popt_1), 'r')
plt.ylabel('p(veh/ft)')
plt.xlabel('V(ft/s)')
plt.title('Greenshields')

# 子图: Van Aerde
plt.subplot(1, 3, 3)
plt.scatter(mat_v.reshape(-1), mat.reshape(-1), s=0.5)
plt.plot(x, Van_Aerde(x, *popt_2), 'r')
plt.ylabel('p(veh/ft)')
plt.xlabel('V(ft/s)')
plt.title('Van Aerde')

plt.show()
