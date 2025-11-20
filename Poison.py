import numpy as np
import fo
import torch
import copy

class Poison_Aggregate:
    def __init__(self, data, epsilon, type="input", ratio=[0.1, 0.8, 0.9, 0.01], protocol='check', c=0.1, delta=None):
        self.data = data.data  # 原始邻接矩阵
        self.type = type  # 投毒类型
        self.ratio = ratio  # [m, s, r, h]
        self.protocol = protocol  # 执行的协议
        self.epsilon = epsilon
        self.c = c
        self.delta = delta
        self.h_user = len(self.data)

    def PA(self):
        self.data = self.poison_data()
        if self.protocol == 'check':
            perturbed_data = self.perturb()
            perturbed_degree = []
        elif self.protocol == 'hybrid':
            perturbed_data, perturbed_degree = self.perturb()
        else:
            raise ValueError("Unsupported protocol!")
        perturbed_data_t = copy.deepcopy(perturbed_data)
        perturbed_degree_t = copy.deepcopy(perturbed_degree)
        results_Imola =[]
        est_degrees, valid_indices, invalid_indices = self.run_protocol(perturbed_data, perturbed_degree)
        results = self.evaluate(est_degrees, valid_indices, invalid_indices)
        est_degrees, valid_indices, invalid_indices = self.run_protocol_t(perturbed_data_t, perturbed_degree_t)
        results_Imola = self.evaluate(est_degrees, valid_indices, invalid_indices)

        return results, results_Imola

    def perturb(self):
        n = len(self.data)
        m_ratio, s_ratio, r_ratio, h_ratio = self.ratio
        rho = 1 / (1 + np.exp(self.epsilon))  # 计算隐私参数 rho
        # 计算恶意用户和目标的数量
        eps_a = self.epsilon * (1 - self.c)
        eps_d = self.epsilon * self.c
        rho_a = 1 / (1 + np.exp(eps_a))  # 计算隐私参数 rho
        num_malicious = int(m_ratio * n / (1 - m_ratio))
        if self.protocol == 'check':
            return fo.rr_perturb(self.data, n, 1 - rho, self.type, num_malicious)
        elif self.protocol == 'hybrid':
            return fo.rr_perturb(self.data, n, 1 - rho_a, self.type, num_malicious), fo.laplace(self.data, eps_d, self.type, num_malicious)
        else:
            raise ValueError("Unsupported protocol!")

    def poison_data(self):
        n = len(self.data)
        m_ratio, s_ratio, r_ratio, h_ratio = self.ratio

        # 计算恶意用户和目标的数量
        num_malicious = int(m_ratio * n/(1-m_ratio))  # 恶意用户数量
        num_targets = int(s_ratio * num_malicious)  # 恶意目标数量
        num_poison_real = int(r_ratio * n)
        num_honest_edge = int(h_ratio * n)
        # 投毒：修改部分边
        poisoned_data = np.zeros((n + num_malicious, n + num_malicious), dtype=int)
        poisoned_data[:n, :n] = self.data

        # 模拟恶意用户的投毒策略（这里简单随机实现）
        malicious_users = range(n, n + num_malicious)  # 新增的恶意用户索引
        # targets = np.random.choice(range(n), num_targets, replace=False)  # 恶意目标（从原始用户中选取）

        for user in malicious_users:
            targets = np.random.choice(range(n, n + num_malicious), num_targets, replace=False)  # 恶意目标
            poison_reals = np.random.choice(range(n), num_poison_real, replace=False)  # 投毒真实用户
            honest_edges = np.random.choice(range(n), num_honest_edge, replace=False)  # 真实边

            poisoned_data[user, targets] = 1
            poisoned_data[targets, user] = 1  # 保持对称性

            # 添加真实边
            poisoned_data[user, honest_edges] = 1
            poisoned_data[honest_edges, user] = 1  # 保持对称性

            # 添加投毒真实边
            poisoned_data[user, poison_reals] = 1

        return poisoned_data

    def run_protocol(self, perturbed_data, perturbed_degree):
        if self.protocol == 'check':
            return self.deg_rr_check(perturbed_data)
        elif self.protocol == 'hybrid':
            return self.deg_hybrid(perturbed_data, perturbed_degree)
        else:
            raise ValueError("Unsupported protocol!")

    def run_protocol_t(self, perturbed_data, perturbed_degree):
        if self.protocol == 'check':
            return self.deg_rr_check_t(perturbed_data)
        elif self.protocol == 'hybrid':
            return self.deg_hybrid_Imola(perturbed_data, perturbed_degree)
        else:
            raise ValueError("Unsupported protocol!")

    def run_protocol_Imola(self, perturbed_data, perturbed_degree):
        if self.protocol == 'check':
            return self.deg_rr_check_Imola(perturbed_data)
        elif self.protocol == 'hybrid':
            return self.deg_hybrid_Imola(perturbed_data, perturbed_degree)
        else:
            raise ValueError("Unsupported protocol!")

    def compute_tau_Imola(self, r, m, n, rho, delta, attack_type="input"):
        """
        计算阈值 tau。
        :param m: 恶意用户数量
        :param n: 总用户数量
        :param rho: 隐私参数
        :param delta: 控制误差范围的参数
        :param attack_type: 攻击类型，"input" 或 "output"
        :return: 计算得到的阈值 tau
        """
        if attack_type == "input":
            tau = m * (1 - 2 * rho) + np.sqrt(m * np.log(4 / delta)) + np.sqrt(3 * n * rho * np.log(4 / delta))
            # tau = n*r*(2*rho-1)**2/2
        elif attack_type == "output":
            tau = m + np.sqrt(3 * n * rho * np.log(2 / delta))
        else:
            raise ValueError("Unsupported attack type! Use 'input' or 'response'.")
        return tau

    def compute_tau_lap_Imola(self, r, m, n, rho, eps_d, delta, attack_type):
        """
        计算阈值 tau。
        :param m: 恶意用户数量
        :param n: 总用户数量
        :param rho: 隐私参数
        :param delta: 控制误差范围的参数
        :param attack_type: 攻击类型，"input" 或 "output"
        :return: 计算得到的阈值 tau
        """
        # _,_,r,_ =self.ratio
        if attack_type == "input":
            tau = m * (1 - 2 * rho) + np.sqrt(m * np.log(8 / delta)) + np.sqrt(3 * n * rho * np.log(8 / delta))

        elif attack_type == "output":
            tau = m + np.sqrt(3 * n * rho * np.log(4 / delta))

        else:
            raise ValueError("Unsupported attack type! Use 'input' or 'response'.")
        return tau
    def deg_rr_check_Imola(self, perturbed_data):
        """
        主函数：先进行度数估计，计算最大度数 d_m，并动态调整 r 和 \tau，
        直至没有超过阈值的节点。

        :param eps: 隐私预算 epsilon
        :param delta: 检测的失败概率
        :return: 最终的估计度数列表
        """
        rho = 1 / (1 + np.exp(self.epsilon))  # 计算隐私参数 rho

        est_degrees, valid_indices, invalid_indices = self.compute_estimated_degrees_Imola(perturbed_data, rho)

        # print("迭代完成，所有节点均满足一致性检查。")
        return est_degrees, valid_indices, invalid_indices

    def deg_hybrid_Imola(self, perturbed_data, perturbed_degree):
        """
        主函数：先进行度数估计，计算最大度数 d_m，并动态调整 r 和 \tau，
        直至没有超过阈值的节点。

        :param eps: 隐私预算 epsilon
        :param delta: 检测的失败概率
        :return: 最终的估计度数列表
        """
        n = len(self.data)  # 用户总数
        eps_a = self.epsilon*(1-self.c)
        rho = 1 / (1 + np.exp(eps_a))  # 计算隐私参数 rho
        eps_d = self.epsilon * self.c

        perturbed_degree = torch.tensor(perturbed_degree, dtype=torch.float32, device=perturbed_data.device)
        est_degrees, valid_indices, invalid_indices = self.compute_estimated_degrees_hybrid_Imola(perturbed_data,
                                                                                            perturbed_degree, rho,
                                                                                            eps_d, self.delta)

        # print("迭代完成，所有节点均满足一致性检查。")
        return est_degrees, valid_indices, invalid_indices

    def compute_estimated_degrees_hybrid_Imola(self, perturbed_data, perturbed_degree, rho, eps_d, delta):
        """
        估计度数并进行一致性检查。

        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值（首次运行时为 None）
        :return: 估计的度数列表，符合/不符合一致性检查的节点索引
        """
        N = perturbed_data.shape[0]
        est_degrees_full = torch.ones(perturbed_data.shape[0], dtype=torch.float32, device=perturbed_data.device)
        valid_indices_full = torch.ones(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)
        invalid_indices_full = torch.zeros(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)

        rows_to_remove = torch.any(torch.isnan(perturbed_data), dim=1)
        perturbed_data = perturbed_data[~rows_to_remove, :]
        perturbed_data = perturbed_data[:, ~rows_to_remove]
        perturbed_degree = perturbed_degree[~rows_to_remove]
        n = perturbed_data.shape[0]  # 更新矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        # 初始化估计度数列表
        est_degrees = (r_11 - rho ** 2 * (n - 1)) / (1 - 2 * rho)

        m,_,r,_ = self.ratio
        tau = self.compute_tau_Imola(r, m, n, rho, 1, self.type)  # 根据 r 计算 tau
        tau_lap = self.compute_tau_lap_Imola(r, m, n, rho, eps_d, delta, self.type)
        # 计算 |r_10 - r_01|
        diff = torch.abs(r_01 - rho * (1 - rho) * (n - 1))

        if tau is None:  # 首次运行时不进行一致性检查
            valid_indices = torch.ones(n, dtype=torch.bool, device=perturbed_data.device)
            invalid_indices = torch.zeros(n, dtype=torch.bool, device=perturbed_data.device)
        else:
            # 满足一致性检查的节点
            valid_indices = (diff <= tau) & (torch.abs(est_degrees-perturbed_degree) <= tau_lap)
            invalid_indices = (diff > tau) | (torch.abs(est_degrees-perturbed_degree) > tau_lap)

        # 对于不一致的节点，返回 NaN
        est_degrees[invalid_indices] = float('nan')
        est_degrees_full[~rows_to_remove] = perturbed_degree
        valid_indices_full[~rows_to_remove] = valid_indices
        invalid_indices_full[~rows_to_remove] = invalid_indices

        return est_degrees_full, valid_indices_full, invalid_indices_full

    def compute_estimated_degrees_Imola(self, perturbed_data, rho):
        """
        使用向量化优化估计度数的计算。
        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值
        :return: 估计的度数列表
        """
        n = perturbed_data.shape[0]  # 矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        # r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        diff = torch.abs(r_01 - rho * (1 - rho) * (n - 1))
        m_ratio, _, r, _ = self.ratio
        m = n*m_ratio
        tau = self.compute_tau_Imola(r, m, n, rho, 1, self.type)
        # 初始化估计度数列表
        est_degrees = torch.empty(n, dtype=torch.float32, device=perturbed_data.device)

        # 满足一致性检查的节点
        valid_indices = diff <= tau
        invalid_indices = diff > tau

        # 计算符合条件的估计度数
        est_degrees[valid_indices] = (r_11[valid_indices] - rho ** 2 * (n - 1)) / (1 - 2 * rho)

        # 对于不一致的节点，返回 None（在 PyTorch 中可用 NaN 表示）
        est_degrees[invalid_indices] = float('nan')

        return est_degrees, valid_indices, invalid_indices
    def compute_tau(self, r, n, rho, delta, attack_type):
        """
        计算阈值 tau。
        :param m: 恶意用户数量
        :param n: 总用户数量
        :param rho: 隐私参数
        :param delta: 控制误差范围的参数
        :param attack_type: 攻击类型，"input" 或 "output"
        :return: 计算得到的阈值 tau
        """
        # _,_,r,_ =self.ratio
        if attack_type == "input":
            # tau = m * abs(1 - 2 * rho) + np.sqrt(m * abs(1 - 2 * rho) * np.log(1 / delta))
            tau = n * r * abs(1 - 2 * rho) / 2
        elif attack_type == "output":
            # tau = m + np.sqrt(m * np.log(2 / delta)) + np.sqrt(3 * n * rho * np.log(2 / delta))
            tau = n * abs(r - rho) / 2
        else:
            raise ValueError("Unsupported attack type! Use 'input' or 'response'.")
        return tau

    def compute_tau_lap(self, r, m, n, rho, eps_d, delta, attack_type):
        """
        计算阈值 tau。
        :param m: 恶意用户数量
        :param n: 总用户数量
        :param rho: 隐私参数
        :param delta: 控制误差范围的参数
        :param attack_type: 攻击类型，"input" 或 "output"
        :return: 计算得到的阈值 tau
        """
        # _,_,r,_ =self.ratio
        if attack_type == "input":
            # tau = m * abs(1 - 2 * rho) + np.sqrt(m * abs(1 - 2 * rho) * np.log(1 / delta))
            # tau = np.sqrt((n-1)/2*np.log(2/delta))+np.abs(m*r*rho/(1-2*rho))-np.log(delta)/eps_d
            tau = np.sqrt((n - 1) / 2 * np.log(2 / delta)) - np.log(delta) / eps_d
        elif attack_type == "output":
            # tau = m + np.sqrt(m * np.log(2 / delta)) + np.sqrt(3 * n * rho * np.log(2 / delta))
            tau = np.sqrt((n-1)/2*np.log(2/delta))+np.abs(m*(r-rho)*rho/(1-2*rho))-np.log(delta)/eps_d
            # tau = np.sqrt((n - 1) / 2 * np.log(2 / delta)) - np.log(delta) / eps_d
        else:
            raise ValueError("Unsupported attack type! Use 'input' or 'response'.")
        return tau

    def compute_tau_t(self, r, m, n, rho, delta, attack_type="input"):
        """
        计算阈值 tau。
        :param m: 恶意用户数量
        :param n: 总用户数量
        :param rho: 隐私参数
        :param delta: 控制误差范围的参数
        :param attack_type: 攻击类型，"input" 或 "output"
        :return: 计算得到的阈值 tau
        """
        if attack_type == "input":
            tau = 0.03
            # tau = n*r*(2*rho-1)**2/2
        elif attack_type == "output":
            tau = 0.1
        else:
            raise ValueError("Unsupported attack type! Use 'input' or 'response'.")
        return tau

    def deg_rr_check_t(self, perturbed_data):
        """
        主函数：先进行度数估计，计算最大度数 d_m，并动态调整 r 和 \tau，
        直至没有超过阈值的节点。

        :param eps: 隐私预算 epsilon
        :param delta: 检测的失败概率
        :return: 最终的估计度数列表
        """
        rho = 1 / (1 + np.exp(self.epsilon))  # 计算隐私参数 rho

        est_degrees, valid_indices, invalid_indices = self.compute_estimated_degrees_t(perturbed_data, rho)

        # print("迭代完成，所有节点均满足一致性检查。")
        return est_degrees, valid_indices, invalid_indices

    def compute_estimated_degrees_t(self, perturbed_data, rho):
        """
        使用向量化优化估计度数的计算。
        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值
        :return: 估计的度数列表
        """
        n = perturbed_data.shape[0]  # 矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        diff = torch.abs((r_01+r_10)/(n-1) - 2*rho * (1 - rho))
        m_ratio, _, r, _ = self.ratio
        m = n*m_ratio
        tau = self.compute_tau_t(r, m, n, rho, 1, self.type)
        # 初始化估计度数列表
        est_degrees = torch.empty(n, dtype=torch.float32, device=perturbed_data.device)

        # 满足一致性检查的节点
        valid_indices = diff <= tau
        invalid_indices = diff > tau

        # 计算符合条件的估计度数
        est_degrees[valid_indices] = (r_11[valid_indices] - rho**2 * (n)) / (1 - 2 * rho)

        # 对于不一致的节点，返回 None（在 PyTorch 中可用 NaN 表示）
        est_degrees[invalid_indices] = float('nan')

        return est_degrees, valid_indices, invalid_indices

    # self.deg_rr_check(perturbed_data, num_malicious)
    def deg_rr_check(self, perturbed_data):
        """
        主函数：先进行度数估计，计算最大度数 d_m，并动态调整 r 和 \tau，
        直至没有超过阈值的节点。

        :param eps: 隐私预算 epsilon
        :param delta: 检测的失败概率
        :return: 最终的估计度数列表
        """

        rho = 1 / (1 + np.exp(self.epsilon))  # 计算隐私参数 rho
        est_degrees = None
        prev_valid_indices = None  # 用于存储上一轮的 valid_indices

        for i in range(1):
            # Step 1: 估计度数，求出最大度数 d_m
            est_degrees, valid_indices, invalid_indices = self.compute_estimated_degrees(perturbed_data, rho, self.type)

            if prev_valid_indices is not None:
                invalid_indices |= ~prev_valid_indices  # 将上一轮 invalid 的节点保持 invalid
                valid_indices &= ~invalid_indices

            if prev_valid_indices is not None and torch.equal(prev_valid_indices, valid_indices):
                break
            prev_valid_indices = valid_indices.clone()  # 更新 prev_valid_indices

            perturbed_data[invalid_indices] = float('nan')  # 剔除不满足一致性的节点
            if torch.all(invalid_indices):
                print("所有节点均不符合一致性要求，停止迭代。")
                break

        # print("迭代完成，所有节点均满足一致性检查。")
        return est_degrees, valid_indices, invalid_indices

    # self.deg_hybrid(perturbed_data, perturbed_degree, num_malicious)
    def deg_hybrid(self, perturbed_data, perturbed_degree):
        """
        主函数：先进行度数估计，计算最大度数 d_m，并动态调整 r 和 \tau，
        直至没有超过阈值的节点。

        :param eps: 隐私预算 epsilon
        :param delta: 检测的失败概率
        :return: 最终的估计度数列表
        """
        n = len(self.data)  # 用户总数
        eps_a = self.epsilon*(1-self.c)
        rho = 1 / (1 + np.exp(eps_a))  # 计算隐私参数 rho
        eps_d = self.epsilon * self.c

        # 初始化变量
        est_degrees = None
        prev_valid_indices = None  # 用于存储上一轮的 valid_indices
        perturbed_degree = torch.tensor(perturbed_degree, dtype=torch.float32, device=perturbed_data.device)
        for i in range(1):
            # Step 1: 估计度数，求出最大度数 d_m
            est_degrees, valid_indices, invalid_indices = self.compute_estimated_degrees_hybrid(perturbed_data, perturbed_degree, rho, eps_d,self.delta, self.type)

            if prev_valid_indices is not None:
                invalid_indices |= ~prev_valid_indices  # 将上一轮 invalid 的节点保持 invalid
                valid_indices &= ~invalid_indices

            if prev_valid_indices is not None and torch.equal(prev_valid_indices, valid_indices):
                break
            prev_valid_indices = valid_indices.clone()  # 更新 prev_valid_indices

            perturbed_data[invalid_indices] = float('nan')  # 剔除不满足一致性的节点
            perturbed_degree[invalid_indices] = float('nan')
            if torch.all(invalid_indices):
                print("所有节点均不符合一致性要求，停止迭代。")
                break

        # print("迭代完成，所有节点均满足一致性检查。")
        return est_degrees, valid_indices, invalid_indices

    def compute_estimated_degrees_hybrid(self, perturbed_data, perturbed_degree, rho, eps_d, delta, type):
        """
        估计度数并进行一致性检查。

        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值（首次运行时为 None）
        :return: 估计的度数列表，符合/不符合一致性检查的节点索引
        """
        N = perturbed_data.shape[0]
        est_degrees_full = torch.ones(perturbed_data.shape[0], dtype=torch.float32, device=perturbed_data.device)
        valid_indices_full = torch.ones(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)
        invalid_indices_full = torch.zeros(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)

        rows_to_remove = torch.any(torch.isnan(perturbed_data), dim=1)
        perturbed_data = perturbed_data[~rows_to_remove, :]
        perturbed_data = perturbed_data[:, ~rows_to_remove]
        perturbed_degree = perturbed_degree[~rows_to_remove]
        n = perturbed_data.shape[0]  # 更新矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        # 初始化估计度数列表
        est_degrees = (r_11 - rho ** 2 * (n - 1)) / (1 - 2 * rho)
        top_k = 10
        r_1 = torch.sum(perturbed_data, dim=1)
        est_degrees1 = (r_1 - rho * (n - 1)) / (1 - 2 * rho)
        if N==n:
            # 输出top-k用户的序号  在naive的方法下计算用户的度数

            # top_k = 10  # 假设我们想要输出前10个用户
            _, top_k_indices = torch.topk(est_degrees1, top_k, largest=True)

            # 输出top-k的用户序号
            # print("Top-{} 用户的序号：".format(top_k), top_k_indices)

            # 计算恶意用户的top-k占比
            num_malicious = int(self.ratio[0] * n )  # 恶意用户数量
            malicious_top_k = torch.sum(top_k_indices >= n-num_malicious).item()
            malicious_top_k_ratio = malicious_top_k / top_k  # 恶意用户占比

            # print("恶意用户在Top-{}中的占比: {:.2f}".format(top_k, malicious_top_k_ratio))

        # 计算tau
        # d_m = torch.max(est_degrees.nan_to_num(float('-inf'))).item()  # 取最大值，忽略 NaN
        top_k_degrees = torch.topk(est_degrees, top_k).values
        if top_k_degrees.numel() > 0:
            if len(self.data) < 40000:
                d_m = torch.max(est_degrees1).item() * 2  # fm
            else:
                d_m = torch.max(est_degrees1).item()*1.5  # fb
        else:
            d_m = n  # 如果 est_degrees 为空，设置最大值为负无穷
        r = d_m / n  # 投毒比例
        tau = self.compute_tau(r, n, rho, None, type)  # 根据 r 计算 tau
        m_ratio = r * 0.2 / (1 + r) / 0.8
        m = n*m_ratio
        tau_lap = self.compute_tau_lap(r, m, n, rho, eps_d, delta, type)
        # 计算 |r_10 - r_01|
        diff = torch.abs(r_10 - r_01)

        if tau is None:  # 首次运行时不进行一致性检查
            valid_indices = torch.ones(n, dtype=torch.bool, device=perturbed_data.device)
            invalid_indices = torch.zeros(n, dtype=torch.bool, device=perturbed_data.device)
        else:
            # 满足一致性检查的节点
            valid_indices = (diff <= tau) & (torch.abs(est_degrees-perturbed_degree) <= tau_lap)
            invalid_indices = (diff > tau) | (torch.abs(est_degrees-perturbed_degree) > tau_lap)

        # 对于不一致的节点，返回 NaN
        est_degrees[invalid_indices] = float('nan')
        est_degrees_full[~rows_to_remove] = perturbed_degree
        valid_indices_full[~rows_to_remove] = valid_indices
        invalid_indices_full[~rows_to_remove] = invalid_indices

        return est_degrees_full, valid_indices_full, invalid_indices_full

    def compute_estimated_degrees(self, perturbed_data, rho, type):
        """
        估计度数并进行一致性检查。

        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值（首次运行时为 None）
        :return: 估计的度数列表，符合/不符合一致性检查的节点索引
        """
        N = perturbed_data.shape[0]
        est_degrees_full = torch.ones(perturbed_data.shape[0], dtype=torch.float32, device=perturbed_data.device)
        valid_indices_full = torch.ones(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)
        invalid_indices_full = torch.zeros(perturbed_data.shape[0], dtype=torch.bool, device=perturbed_data.device)

        rows_to_remove = torch.any(torch.isnan(perturbed_data), dim=1)
        perturbed_data = perturbed_data[~rows_to_remove, :]
        perturbed_data = perturbed_data[:, ~rows_to_remove]

        n = perturbed_data.shape[0]  # 更新矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        # 初始化估计度数列表
        est_degrees = (r_11 - rho ** 2 * (n - 1)) / (1 - 2 * rho)
        top_k = 10

        r_1 = torch.sum(perturbed_data, dim=1)
        est_degrees1 = (r_1 - rho * (n - 1)) / (1 - 2 * rho)

        if N==n:
            # 输出top-k用户的序号  在naive的方法下计算用户的度数

            # top_k = 10  # 假设我们想要输出前10个用户
            a, top_k_indices = torch.topk(est_degrees1, top_k, largest=True)

            # 输出top-k的用户序号
            # print("Top-{} 用户的序号：".format(top_k), top_k_indices)

            # 计算恶意用户的top-k占比
            num_malicious = int(self.ratio[0] * n )  # 恶意用户数量
            malicious_top_k = torch.sum(top_k_indices >= n-num_malicious).item()
            malicious_top_k_ratio = malicious_top_k / top_k  # 恶意用户占比

            # print("恶意用户在Top-{}中的占比: {:.2f}".format(top_k, malicious_top_k_ratio))

        # 计算tau
        # d_m = torch.max(est_degrees.nan_to_num(float('-inf'))).item()  # 取最大值，忽略 NaN
        top_k_degrees = torch.topk(est_degrees, top_k).values
        if est_degrees.numel() > 0:
            if len(self.data) < 40000:
                d_m = torch.mean(top_k_degrees).item() * 1.5 # fm
            else:
                d_m = torch.max(est_degrees).item()*(rho+0.3) # fb
        else:
            d_m = n  # 如果 est_degrees 为空，设置最大值为负无穷
        r = d_m / n  # 投毒比例
        tau = self.compute_tau(r, n, rho, None, type)  # 根据 r 计算 tau

        # 计算 |r_10 - r_01|
        diff = torch.abs(r_10 - r_01)

        if tau is None:  # 首次运行时不进行一致性检查
            valid_indices = torch.ones(n, dtype=torch.bool, device=perturbed_data.device)
            invalid_indices = torch.zeros(n, dtype=torch.bool, device=perturbed_data.device)
        else:
            # 满足一致性检查的节点
            valid_indices = diff <= tau
            invalid_indices = diff > tau

        # 对于不一致的节点，返回 NaN
        est_degrees[invalid_indices] = float('nan')
        est_degrees_full[~rows_to_remove] = est_degrees
        valid_indices_full[~rows_to_remove] = valid_indices
        invalid_indices_full[~rows_to_remove] = invalid_indices

        return est_degrees_full, valid_indices_full, invalid_indices_full
    def compute_estimated_degrees0(self, perturbed_data, rho, tau):
        """
        使用向量化优化估计度数的计算。
        :param perturbed_data: 扰动后的邻接矩阵 (torch.Tensor)
        :param rho: 隐私参数相关值
        :param tau: 阈值
        :return: 估计的度数列表
        """
        n = perturbed_data.shape[0]  # 矩阵大小

        # 使用矩阵运算计算 r_11、r_01 和 r_10
        r_11 = torch.sum(perturbed_data * perturbed_data.T, dim=1)
        r_01 = torch.sum((1 - perturbed_data) * perturbed_data.T, dim=1)
        # r_10 = torch.sum(perturbed_data * (1 - perturbed_data.T), dim=1)

        # 计算 |r_10 - r_01|
        diff = torch.abs(r_01 - rho * (1 - rho) * (n - 1))


        # 初始化估计度数列表
        est_degrees = torch.empty(n, dtype=torch.float32, device=perturbed_data.device)

        # 满足一致性检查的节点
        valid_indices = diff <= tau
        invalid_indices = diff > tau

        # 计算符合条件的估计度数
        est_degrees[valid_indices] = (r_11[valid_indices] - rho ** 2 * (n - 1)) / (1 - 2 * rho)

        # 对于不一致的节点，返回 None（在 PyTorch 中可用 NaN 表示）
        est_degrees[invalid_indices] = float('nan')

        return est_degrees, valid_indices, invalid_indices
    def evaluate(self, est_degrees, valid_indices, invalid_indices):
        """
        计算最大误差、假阴性率、假阳性率，以及精确度、召回率和 F1 分数。
        :param est_degrees: 估计的度数
        :param threshold: 判定用户是否为恶意的阈值
        :return: 最大误差、假阴性率、假阳性率、精确度、召回率、F1 分数
        """
        n = self.h_user  # 诚实用户数量
        m = len(self.data) - n  # 恶意用户数量

        if isinstance(est_degrees, torch.Tensor):
            est_degrees = est_degrees.cpu().numpy()
        if isinstance(valid_indices, torch.Tensor):
            valid_indices = valid_indices.cpu().numpy()
        if isinstance(invalid_indices, torch.Tensor):
            invalid_indices = invalid_indices.cpu().numpy()

        # 计算真实度数
        real_degrees = np.sum(self.data, axis=1)

        honest_mask = ~invalid_indices  # 前 n 个用户为诚实用户
        malicious_mask = invalid_indices  # 后 m 个用户为恶意用户

        # 诚实用户集合和恶意用户集合的布尔掩码

        # 最大误差计算
        honest_errors = np.abs(est_degrees[honest_mask] - real_degrees[honest_mask])
        max_honest_error = np.nanmax(honest_errors) if len(honest_errors) > 0 else None

        malicious_errors = np.abs(est_degrees[malicious_mask] - real_degrees[malicious_mask])
        max_malicious_error = np.nanmax(malicious_errors) if len(malicious_errors) > 0 else None

        honest_mask = np.arange(len(self.data)) < n  # 前 n 个用户为诚实用户
        malicious_mask = np.arange(len(self.data)) >= n  # 后 m 个用户为恶意用户

        # 假阳性 (诚实用户被误分类为恶意用户)
        false_positives = np.sum(invalid_indices & honest_mask)

        # 假阴性 (恶意用户被误分类为诚实用户)
        false_negatives = np.sum(valid_indices & malicious_mask)

        # 真正例 (恶意用户被正确分类为恶意用户)
        true_positives = np.sum(invalid_indices & malicious_mask)

        # 计算假阴性率 (FNR) 和假阳性率 (FPR)
        fnr = false_negatives / m if m > 0 else None
        fpr = false_positives / n if n > 0 else None
        # 计算精确度 (Precision)
        precision = true_positives / (true_positives + false_positives) if (
                                                                                       true_positives + false_positives) > 0 else None

        # 计算召回率 (Recall)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else None

        # 计算 F1 分数
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )

        # print("evaluate")
        return max_honest_error, max_malicious_error, fnr, fpr, precision, recall, f1_score


