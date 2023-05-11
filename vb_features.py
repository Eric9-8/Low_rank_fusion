# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/5/9 9:00
import numpy as np
import pandas as pd
import math
from scipy.fftpack import fft

# path = r'D:\Mobilenet_CoordinateAttention_GAF\Fusion_net\The winter\服务器上的代码\DRSN\dataset\6_all_data.csv'
root = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\Vibration_labeled.csv'
root1 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\vb_.csv'
root2 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\imf1.csv'
root3 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\imf2.csv'
root4 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\vb_data_4200.csv'
root5 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\imf1_4200.csv'
root6 = 'D:\\Pytorch\\Fusion_lowrank\\data\\csv_data\\imf2_4200.csv'
number = 1050


def time_features(dataset):
    data = dataset.transpose()
    data_feature = data.describe()
    data_feature = data_feature.transpose()

    df_mean = data.mean()  # 均值
    df_std = data.std()  # 标准差
    df_var = data.var()  # 方差
    df_rms = (df_mean ** 2 + df_std ** 2).apply(math.sqrt)  # 均方根
    df_skew = data.skew()  # 偏度
    df_kurt = data.kurt()  # 峭度
    df_bx = df_rms / (abs(data).mean())  # 波形因子
    df_fz = data_feature['max'] / df_rms  # 峰值因子
    df_mc = data_feature['max'] / (abs(data).mean())  # 脉冲因子
    df_yd = data_feature['max'] / ((np.sqrt(abs(data))).mean()) ** 2  # 裕度
    feature_data = pd.concat([df_mean, df_std, df_var, df_rms, df_skew, df_kurt, df_bx, df_fz, df_mc, df_yd], axis=1)
    feature_data.index = range(len(feature_data))
    # feature_data.columns = (['mean', 'std', 'var', 'rms', 'skew', 'kurt', 'bx', 'fz', 'mc', 'yd'])
    # feature_data.columns = (['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8', 'FT9', 'FT10'])
    feature_data.columns = (
        ['FT1_imf1', 'FT2_imf1', 'FT3_imf1', 'FT4_imf1', 'FT5_imf1', 'FT6_imf1', 'FT7_imf1', 'FT8_imf1', 'FT9_imf1',
         'FT10_imf1'])

    return feature_data


def freq_feature(dataset):
    fs = 12800
    # Ts = 1 / fs
    N = 6400
    # t = [i * Ts for i in range(N)]
    # n = 1000
    f = np.array([i / fs for i in range(N)]).reshape(6400, 1)
    K = N / 2.56

    data = dataset.transpose()
    data = fft(data)
    data = abs(data)
    FF1 = sum(data) / K
    # FF1 = data.mean()
    FF2 = sum(f * data) / sum(data)
    FF3 = np.sqrt(sum([j * j for j in f] * data) / sum(data))
    FF4 = np.sqrt(sum([k * k for k in (f - FF2)] * data) / K)
    FF5 = np.sqrt(sum([l ** 4 for l in f] * data) / sum([l * l for l in f] * data))
    FF6 = sum([m * m for m in f] * data) / np.sqrt(sum(data) * sum([m ** 4 for m in f] * data))
    FF7 = FF4 / FF1
    FF8 = sum([n ** 3 for n in (f - FF2)] * data) / np.array([K * (n ** 3) for n in FF4])
    FF9 = sum([n ** 4 for n in (f - FF2)] * data) / np.array([K * (n ** 3) for n in FF4])
    FF10 = sum(np.sqrt(abs(f - FF2)) * data) / (K * np.sqrt(FF4))
    lst_feature = [FF1, FF2, FF3, FF4, FF5, FF6, FF7, FF8, FF9, FF10]
    feature_data = pd.DataFrame(lst_feature).transpose()
    feature_data.index = range(len(feature_data))
    # feature_data.columns = (['FF1', 'FF2', 'FF3', 'FF4', 'FF5', 'FF6', 'FF7', 'FF8', 'FF9', 'FF10'])
    feature_data.columns = (
        ['FF1_imf2', 'FF2_imf2', 'FF3_imf2', 'FF4_imf2', 'FF5_imf2', 'FF6_imf2', 'FF7_imf2', 'FF8_imf2', 'FF9_imf2',
         'FF10_imf2'])
    # print(FF1)
    return feature_data


def pd_process(path):
    data = pd.read_csv(path)
    data_set = data.iloc[:, :]
    tf = time_features(data_set)
    ff = freq_feature(data_set)
    feature_data = pd.concat([tf, ff], axis=1)
    feature_data['label'] = ''
    feature_data = feature_data.copy()

    feature_data.loc[0:number, 'label'] = 0
    feature_data.loc[number:number * 2, 'label'] = 1
    feature_data.loc[number * 2:number * 3, 'label'] = 2
    feature_data.loc[number * 3:number * 4, 'label'] = 3

    return feature_data


if __name__ == "__main__":
    fd = pd_process(root6)
    fd.to_csv(r"D:\Pytorch\Fusion\data\csv_data\imf2_4200_feature.csv", index=False, sep=',')
