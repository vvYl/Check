import numpy as np
import xxhash
import torch
import time

def laplace_mechanism(scale):
    """Generate Laplace noise."""
    return np.random.laplace(0, scale)

def laplace(real_dist, eps, type, m):
    real_de = np.sum(real_dist, axis=1)
    # 每一行求和
    domain = len(real_de)
    noisy_de = real_de
    if type == "input":
        # 对所有用户添加噪声
        for i in range(domain):
            noisy_de[i] += laplace_mechanism(1 / eps)
    else:
        # 对前 domain - m 个用户添加噪声
        for i in range(domain - m):
            noisy_de[i] += laplace_mechanism(1 / eps)
        # 对最后 m 个用户保持原始度数
        # noisy_de[domain - m:] = real_de[domain - m:]

    return noisy_de

def rr(real_dist, eps):
    domain = len(real_dist)
    ee = np.exp(eps)

    p = ee / (ee + domain - 1)
    q = 1 / (ee + domain - 1)

    noisy_samples = rr_perturb(real_dist, domain, p)
    est_dist = rr_aggregate(noisy_samples, domain, p, q)

    return est_dist


# def rr_perturb(real_dist, domain, p):
#     n = domain
#     noisy_samples = np.zeros_like(real_dist)
#
#     for i in range(n):
#         for j in range(n):
#             if i != j:  # 跳过自连接
#                 # 对每一对用户(i, j)进行扰动
#                 if real_dist[i, j] == 1:
#                     # 保持原样的概率为p，翻转的概率为1-p
#                     noisy_samples[i, j] = np.random.choice([1, 0], p=[p, 1-p])
#                 else:
#                     noisy_samples[i, j] = np.random.choice([1, 0], p=[1-p, p])
#
#     return noisy_samples


def rr_perturb(real_dist, domain, p ,type, m):
    """
    对邻接矩阵 real_dist 使用随机响应机制进行扰动。
    使用 torch.bernoulli 优化原始实现。
    :param real_dist: 原始邻接矩阵 (torch.Tensor)
    :param domain: 矩阵大小
    :param p: 保持原样的概率
    :return: 扰动后的邻接矩阵 (torch.Tensor)
    """
    # 将 real_dist 转换为 torch.Tensor（如果不是的话）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.perf_counter()
    if not isinstance(real_dist, torch.Tensor):
        real_dist = torch.tensor(real_dist, dtype=torch.float32).to(device)
    real_dist_conversion_time = time.perf_counter() - start_time
    # print(f"Step 1: Convert real_dist to tensor -> Time taken: {real_dist_conversion_time:.6f} seconds")

    # 生成随机矩阵，元素以概率 p 为 1
    start_time = time.perf_counter()
    random_matrix = torch.bernoulli(torch.full((domain, domain), 1-p, device=device))
    random_matrix_generation_time = time.perf_counter() - start_time
    # print(f"Step 2: Generate random matrix -> Time taken: {random_matrix_generation_time:.6f} seconds")

    # 执行随机扰动：对于每个元素，根据概率决定是否翻转
    noisy_samples = (real_dist + random_matrix) % 2

    # 确保对角线元素为 0（无自环）
    noisy_samples.fill_diagonal_(0)
    if type == "output" and m > 0:
        # 将最后 m 个用户的数据恢复为原始数据
        noisy_samples[-m:, :] = real_dist[-m:, :]

    return noisy_samples


def rr_aggregate(noisy_samples, domain, p, q):
    n = domain
    estimated_degrees = np.zeros(n)

    # 聚合过程：计算每个用户的度估计值
    for i in range(n):
        # 对每个用户i，计算其邻接矩阵中的总和
        estimated_degrees[i] = np.sum(noisy_samples[i])

    # 根据校正公式进行校正
    a = 1.0 / (p - q)
    b = n * q / (p - q)

    est_degrees_corrected = a * estimated_degrees - b

    return est_degrees_corrected
