from data import Data
import numpy as np
from Poison import Poison_Aggregate
import torch
from scipy.optimize import fsolve

def equation(r, rmr, t):
    return (r**2 + r) * rmr + r - t

def main():
    # 加载数据
    epsilon_values = [0.5,1,1.5,2.0,2.5,3.0,3.5,4.0]
    rmr = 0.2  # malicious/honest
    t = 0.15 #fb
    # ratio:m:恶意用户数量 s：恶意目标数量 r：投毒真实用户 h：允许存在一定的真实边
    # 使用 fsolve 求解方程
    r_initial_guess = 0  # 设定初始猜测值
    r = fsolve(equation, r_initial_guess, args=(rmr, t))[0]

    s = 0.8
    m = r * rmr / (1 + r) / s
    # for top_k in K:
    for epsilon in epsilon_values:
        # 设置运行次数
        print(f"Running experiment with epsilon = {epsilon}")
        data = Data(dataname='facebook_combined', limit=88243)
        # data = Data(dataname='lastfm', limit=27806)
        cor = []
        sou = []
        pre = []
        rec = []
        f1 = []
        cor_I = []
        sou_I = []
        pre_I = []
        rec_I = []
        f1_I = []

        num_runs = 1
        for run in range(num_runs):

            poison_data = Poison_Aggregate(data, epsilon, type="input", ratio=[m, s, r, 0.001], protocol='check')
            # poison_data = Poison_Aggregate(data, epsilon, type="input", ratio=[m, s, r, 0.001], protocol='hybrid',c=0.3, delta=0.0000001)
            # poison_data = Poison_Aggregate(data, epsilon, type="output", ratio=[m, s, r, 0.001], protocol='hybrid', c=0.3,delta=0.0000001)
            results, results_Imola = poison_data.PA()
            correctness, soundness, fnr, fpr, precision, recall, f1_score = results
            cor.append(correctness)
            sou.append(soundness)
            pre.append(precision)
            rec.append(recall)
            f1.append(f1_score)
            correctness, soundness, fnr, fpr, precision, recall, f1_score = results_Imola
            cor_I.append(correctness)
            sou_I.append(soundness)
            pre_I.append(precision)
            rec_I.append(recall)
            f1_I.append(f1_score)
        print(f"Ours ave-correctness, soundness: {safe_mean(cor), safe_mean(sou), safe_mean(pre), safe_mean(rec), safe_mean(f1)}")
        print(
            f"Imola's ave-correctness, soundness: {safe_mean(cor_I), safe_mean(sou_I), safe_mean(pre_I), safe_mean(rec_I), safe_mean(f1_I)}")


def safe_mean(data):
    """
    安全计算均值，过滤掉 None 或 NaN 值。
    :param data: 输入数据 (list, numpy array, torch.Tensor)
    :return: 均值 (float) 或 None（如果全为 None 或 NaN）
    """
    # 如果是 PyTorch Tensor，先转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 如果数据是 None 或空，直接返回 None
    if data is None or len(data) == 0:
        return None

    # 将数据转为 NumPy 数组，并强制转换为 float 类型，将 None 转为 NaN
    data = np.array(data, dtype=float)

    # 过滤掉 NaN 值
    filtered_data = data[~np.isnan(data)]

    # 如果过滤后为空，返回 None
    if len(filtered_data) == 0:
        return None

    # 计算均值
    return np.mean(filtered_data)

main()
