import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time
import os

# 配置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DroneTSP:
    def __init__(self, matrix):
        """
        初始化 TSP 问题求解器
        二维列表或 numpy 数组，表示带权无向完全图的邻接矩阵
        """
        if isinstance(matrix, np.ndarray):
            self.adj_matrix = matrix
        else:
            self.adj_matrix = np.array(matrix, dtype=float)

        self.n = len(self.adj_matrix)

        # 检查矩阵是否为方阵
        if self.adj_matrix.shape[0] != self.adj_matrix.shape[1]:
            raise ValueError("输入的必须是 N x N 的方阵")

    def visualize(self, path=None, title="无人机巡检路线图"):
        """可视化图和路径"""
        plt.figure(figsize=(10, 8))
        G = nx.Graph()

        # 添加节点
        for i in range(self.n):
            G.add_node(i)

        pos = nx.spring_layout(G, seed=42, k=0.5)

        # 1. 画所有节点
        nx.draw_networkx_nodes(G, pos, node_color='#A0CBE2', node_size=600, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

        # 2. 如果有路径，画路径边
        if path:
            # 生成路径边列表 [(0,1), (1,3)...]
            path_edges = list(zip(path, path[1:]))

            # 画路径实线
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.5, alpha=0.8)

            # 在边上标注权重
            edge_labels = {}
            total_weight = 0
            for u, v in path_edges:
                weight = self.adj_matrix[u][v]
                edge_labels[(u, v)] = f"{weight:.1f}"
                total_weight += weight

            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

            title += f"\n总路径长度: {total_weight:.2f}"

        else:
            pass

        plt.title(title)
        plt.axis('off')
        plt.show()

    def solve_exact(self):
        """
        精确解：暴力枚举
        时间复杂度 O(N!)
        """
        print(f"\n正在计算 {self.n} 个节点的精确路径...")
        start_time = time.time()

        min_path = None
        min_dist = float('inf')

        # 固定起点为0，生成 1 到 n-1 的全排列
        vertex_indices = list(range(1, self.n))

        for perm in itertools.permutations(vertex_indices):
            # 构造路径：0 -> p1 -> ... -> pn -> 0
            current_path = [0] + list(perm) + [0]

            current_dist = 0

            # 计算路径长度
            for i in range(len(current_path) - 1):
                u, v = current_path[i], current_path[i+1]
                w = self.adj_matrix[u][v]
                # 矩阵中0或负数表示不通
                if w <= 0 and u != v:
                    pass
                current_dist += w

            if current_dist < min_dist:
                min_dist = current_dist
                min_path = current_path

        end_time = time.time()
        print(f"精确解耗时: {end_time - start_time:.4f} 秒")
        return min_path, min_dist

    def solve_approx(self):
        """
        近似解：最近邻算法
        时间复杂度 O(N^2)
        """
        print(f"\n正在计算 {self.n} 个节点的近似路径...")
        start_time = time.time()

        visited = [False] * self.n
        current_node = 0
        path = [0]
        visited[0] = True
        total_dist = 0

        for _ in range(self.n - 1):
            nearest_dist = float('inf')
            nearest_node = -1

            # 寻找最近的邻居
            for next_node in range(self.n):
                if not visited[next_node]:
                    dist = self.adj_matrix[current_node][next_node]
                    # 距离必须大于0（排除自身）且更小
                    if 0 < dist < nearest_dist:
                        nearest_dist = dist
                        nearest_node = next_node

            if nearest_node == -1:
                print("错误：图不连通，无法找到回路")
                return None, 0

            visited[nearest_node] = True
            path.append(nearest_node)
            total_dist += nearest_dist
            current_node = nearest_node

        # 回到起点
        return_dist = self.adj_matrix[current_node][0]
        total_dist += return_dist
        path.append(0)

        end_time = time.time()
        print(f"近似解耗时: {end_time - start_time:.4f} 秒")
        return path, total_dist

    @staticmethod
    def from_input():
        """
        仿照参考代码实现的输入接口
        """
        while True:
            try:
                method = int(input("请选择输入邻接矩阵的方式 (0: 手动输入, 1: 文件输入): "))
                if method in [0, 1]:
                    break
                print("输入错误，请输入 0 或 1")
            except ValueError:
                print("请输入数字")

        matrix = []
        n = 0

        if method == 1:
            # 文件输入
            filename = input("请输入文件名 (例如 test.txt): ")
            try:
                with open(filename, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                # 第一行读取节点数 N
                n = int(lines[0])

                # 读取接下来的 N 行作为矩阵
                if len(lines) < n + 1:
                    raise ValueError("文件行数不足")

                print(f"检测到 {n} 个节点，正在读取矩阵...")

                for i in range(1, n + 1):
                    row_elements = list(map(float, lines[i].split()))
                    if len(row_elements) != n:
                        raise ValueError(f"第 {i} 行数据列数不正确，应为 {n} 列")
                    matrix.append(row_elements)

            except FileNotFoundError:
                print(f"错误：找不到文件 {filename}")
                return None
            except Exception as e:
                print(f"文件读取错误: {e}")
                return None

        else:
            # 手动输入
            while True:
                try:
                    n = int(input("请输入检测点(节点)数量 N: "))
                    if n > 1:
                        break
                    print("节点数必须大于 1")
                except ValueError:
                    print("请输入整数")

            print(f"请输入邻接矩阵 (共 {n} 行，每行 {n} 个数字，空格分隔):")
            print("提示：主对角线(自己到自己)通常为0")

            for i in range(n):
                while True:
                    try:
                        line = input(f"请输入第 {i + 1} 行: ")
                        row_elements = list(map(float, line.split()))
                        if len(row_elements) != n:
                            print(f"错误：该行应当有 {n} 个数字，当前有 {len(row_elements)} 个")
                            continue
                        matrix.append(row_elements)
                        break
                    except ValueError:
                        print("输入包含非数字字符，请重试")

        return DroneTSP(matrix)

if __name__ == "__main__":
    # 获取矩阵
    tsp_solver = DroneTSP.from_input()

    if tsp_solver:
        # 打印矩阵供检查
        print("\n当前输入的邻接矩阵:")
        print(np.array(tsp_solver.adj_matrix))

        n = tsp_solver.n

        # 规模较小，精确解和近似解都跑
        if n < 7:
            path_exact, dist_exact = tsp_solver.solve_exact()
            print(f"精确最优路径: {path_exact}")
            print(f"最短距离: {dist_exact:.2f}")
            tsp_solver.visualize(path_exact, title=f"精确解 (N={n}) - 暴力枚举")

        # 规模较大跑近似解
        else:
            print(f"\n警告：节点数 N={n} 较大，跳过精确解计算 (耗时过长)，仅计算近似解。")
            path_approx, dist_approx = tsp_solver.solve_approx()
            print(f"近似路径: {path_approx}")
            print(f"近似距离: {dist_approx:.2f}")
            tsp_solver.visualize(path_approx, title=f"近似解 (N={n}) - 最近邻算法")