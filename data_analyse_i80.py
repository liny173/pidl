import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
#     mape=(abs(y_predict -y_test)/ y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    mape = []
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i, j] != 0:
                mape.append(abs(y_predict[i, j]-y_test[i, j])/y_test[i, j])
    return mae, rmse, np.mean(mape), r_2  # mse


# 获取7个模型在4种抽样率下的指标
def model_id(list):
    model_ident = []
    for i in range(7):
        # 7个模型
        mi = []
        for j in range(4):
            # 4种抽样率
            mi.append(list[j][i])
        model_ident.append(mi)
    return model_ident


# 绘多折线图
def model_draw(list, lab):
    for k in range(len(list)):
        plt.plot(list[k], label=labels_1[k], color=colors_1[k], marker=markers[k], markersize=8)

    plt.xticks([0, 1, 2, 3], ['33%', '20%', '14%', '11%'], fontsize=8)
    plt.xlabel('Observation Data Occupation', fontsize=10)
    plt.ylabel(f'{lab}', fontsize=10)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.title(f'{lab} under Different Observation Data Occupation', fontsize=12)


# 全局绘图的信息
# 标签
labels_1 = ['Huang et al.', 'LWR+Greenshields', 'LWR+Greenberg', 'LWR+Van Aerde',
            'Ross+Greenshields', 'Ross+Greenberg', 'Ross+Van Aerde']
# 颜色
colors = ['#d6e4d2', '#c1d5be', '#9ec3aa', '#76b5af', '#3b85b6', '#2c67a7', '#274b83']
colors_1 = ['#e97123', '#d584be', '#f5e5b3', '#c1dbab', '#00b587', '#29719c', '#b4c8e1']
# 标记
markers = ['*', '^', '<', '>', '+', 'o', 'x']

# o5: 20%的抽样率 o3: 33%的抽样率 o7: 14%的抽样率 o9: 11%的抽样率
cs = ['o3', 'o5', 'o7', 'o9']

# 存储各抽样率下，每种模型的MAE, RMSE, MAPE, R2
MAEs = []
RMSEs = []
MAPEs = []
R2s = []

# 在4种抽样率下循环获取数据
for c in cs:
    print(f'抽样{c}:')
    print('-' * 20)
    # 数据导入
    origin = np.load(r'D:\01\traffic\pidl\PIDL\新建文件夹\data\mat_v_i80.npz')['data']
    LGS10_5 = np.load(
        f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\LWRGreenshields_dense10_estimation_skyeye_{c}.npz')[
        'data']
    LGS2_5 = \
    np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\LWRGreenshields_2lstm_estimation_skyeye_{c}.npz')[
        'data']
    LGG2_5 = \
    np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\LWRGreenberg_2lstm_estimation_skyeye_{c}.npz')[
        'data']
    LV2_5 = \
    np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\LWRVanAerde_dense10_estimation_skyeye_{c}.npz')[
        'data']
    RGS2_5 = \
    np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\ROSSGreenshields_estimation_skyeye_{c}.npz')[
        'data']
    RGG2_5 = \
    np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\ROSSGreenberg_estimation_skyeye_{c}.npz')['data']
    RV2_5 = np.load(f'D:\\01\\traffic\pidl\PIDL\新建文件夹\\res_data\\res\i80\ROSSVanAerde_estimation_skyeye_{c}.npz')[
        'data']
    # 转化和清洗
    origin = pd.DataFrame(origin)
    origin = origin.fillna(0)
    origin = np.array(origin)

    plot_fig = [origin, LGS10_5, LGS2_5, LGG2_5, LV2_5, RGS2_5, RGG2_5, RV2_5]
    # 极值
    min_v = 0
    max_v = 0
    for i in plot_fig:
        min_t = np.min(i)
        max_t = np.max(i)
        if min_v > min_t:
            min_v = min_t
        if max_t > max_v:
            max_v = max_t

    print('max_v:', max_v)
    print('min_v:', min_v)

    labels = ['Origin', 'LWR+Greenshields+dense', 'LWR+Greenshields', 'LWR+Greenberg', 'LWR+Van Aerde',
              'Ross+Greenshields', 'Ross+Greenberg', 'Ross+Van Aerde']

    for i in range(len(labels)):
        print('-' * 10 * i)
        print(labels[i])
        re = evaluation(origin, plot_fig[i])
        print('MAE:', re[0])
        print('RMSE:', re[1])
        print('MAPE:', re[2])
        print('R2:', re[3])

    max_v = 50
    min_v = 0

    num = 1
    # 鸟瞰图
    plt.figure(figsize=(20, 12))
    for i in plot_fig:
        plt.subplot(2, 4, num)

        ax = sns.heatmap(i, vmin=min_v, vmax=max_v, cmap='Blues')
        #     i = (i-min_v)/(max_v-min_v)
        #     sns.heatmap(i,vmin = 0,vmax = 1,cmap='Blues')
        plt.xticks([])
        plt.yticks([])
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        plt.title(labels[num - 1], fontsize=12)
        plt.subplots_adjust(hspace=0.2)
        plt.subplots_adjust(wspace=0.3)
        num = num + 1

    # plt.show()

    MAE_5 = []
    RMSE_5 = []
    MAPE_5 = []
    R2_5 = []
    plt.figure(figsize=(12, 5))
    for i in plot_fig[1:]:
        mae, rmse, mape, r_2 = evaluation(plot_fig[0], i)
        MAE_5.append(mae)
        RMSE_5.append(rmse)
        MAPE_5.append(mape)
        R2_5.append(r_2)
    # plt.figure(figsize = (10,20))
    MAEs.append(MAE_5)
    RMSEs.append(RMSE_5)
    MAPEs.append(MAPE_5)
    R2s.append(R2_5)

    # Linechart
    plt.subplot(1, 2, 1)
    plt.plot(MAE_5, label='MAE', marker='*', markersize=8, color='#e1c855')
    plt.plot(RMSE_5, label='RMSE', marker='<', markersize=8, color='#e07b54')
    # plt.plot(MAPE_5,label = 'MAPE',marker = '>',markersize=15)
    plt.plot(R2_5, label='R2', marker='o', markersize=8, color='#51b1b7')
    rate = round(1.0 / int(c[1]), 2) * 100
    plt.title(f'NGSIM i80 {round(rate)}% data', fontsize=12)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.xticks([0, 1, 2, 3, 4, 5, 6], labels=labels_1, fontsize=8, rotation=30)
    plt.legend(loc='upper right', fontsize=8)

    # Barchart
    plt.subplot(1, 2, 2)
    plt.bar([0, 1, 2, 3, 4, 5, 6], MAPE_5, color=colors)  # hatch=['/','|','-','*','+','o','.']
    plt.xticks([0, 1, 2, 3, 4, 5, 6], labels=labels_1, fontsize=8, rotation=30)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('MAPE (%)', fontsize=10)
    plt.title(f'NGSIM i80 {round(rate)}% data', fontsize=12)
    plt.tight_layout()
    # plt.show()


# 获取指标
model_mae = model_id(MAEs)
model_rmse = model_id(RMSEs)
model_mape = model_id(MAPEs)
model_r2 = model_id(R2s)

# 各个模型指标的绘图
plt.figure(figsize=(8, 6))
model_draw(model_mae, 'MAE')

plt.figure(figsize=(8, 6))
model_draw(model_rmse, 'RMSE')

plt.figure(figsize=(8, 6))
model_draw(model_mape, 'MAPE')

plt.figure(figsize=(8, 6))
model_draw(model_r2, 'R2')

plt.show()
