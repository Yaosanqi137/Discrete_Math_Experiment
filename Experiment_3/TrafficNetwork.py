import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrafficAnalyzer:
    def __init__(self, matrix):
        self.adj_matrix = np.array(matrix, dtype=int)
        self.n = len(self.adj_matrix)

        # 自动标定边 e1, e2, ... (按行顺序标定)
        self.edge_list = []  # 存储 (u, v, label)
        self.edge_map = {}   # 存储 (u, v) -> label
        edge_count = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] == 1:
                    label = f"e{edge_count}"
                    self.edge_list.append((i, j, label))
                    self.edge_map[tuple(sorted((i, j)))] = label
                    edge_count += 1

        # 构建 NetworkX 图对象用于计算
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        for u, v, label in self.edge_list:
            self.G.add_edge(u, v, label=label)

    def solve_spanning_trees(self):
        """
        任务3：计算生成树总个数 (矩阵树定理)
        """
        if not nx.is_connected(self.G):
            return 0

        # 度矩阵 D
        D = np.diag(np.sum(self.adj_matrix, axis=1))
        # 拉普拉斯矩阵 L = D - A
        L = D - self.adj_matrix
        # 删去最后一项，计算余子式
        reduced_L = L[:-1, :-1]
        count = round(np.linalg.det(reduced_L))
        return abs(count)

    def get_one_spanning_tree(self):
        """
        任务2：构建一颗生成树并返回其邻接矩阵
        """
        T = nx.minimum_spanning_tree(self.G)
        T_adj = nx.to_numpy_array(T, weight=None, dtype=int)

        # 提取属于树的边标签
        tree_edges = []
        for u, v in T.edges():
            tree_edges.append(self.edge_map[tuple(sorted((u, v)))])

        return T_adj, T, tree_edges

    def solve_basic_cycles(self, T):
        """
        任务4：求解基本回路系统 (最小绕行单元库)
        """
        basic_cycles = []
        # 找出不在生成树 T 中的边 (连枝/余树边)
        for u, v, label in self.edge_list:
            if not T.has_edge(u, v):
                # 在生成树中寻找 u 到 v 的唯一路径
                path = nx.shortest_path(T, source=u, target=v)
                cycle_edges = []
                # 路径上的边
                for k in range(len(path) - 1):
                    e_label = self.edge_map[tuple(sorted((path[k], path[k+1])))]
                    cycle_edges.append(e_label)
                # 加上当前这条余树边
                cycle_edges.append(label)
                # 排序整理标签 e1e2e3
                cycle_edges.sort(key=lambda x: int(x[1:]))
                basic_cycles.append(cycle_edges)

        # 格式化输出字符串
        formatted_cycles = ["".join(c) for c in basic_cycles]
        return basic_cycles, formatted_cycles

    def solve_cycle_space(self, basic_cycles_list):
        """
        任务5：求解环路空间
        """
        # 环路空间是基本回路组的幂集，通过对称差(XOR)运算得到
        cycle_space = [set()] # 初始包含空环 Φ

        # 转换基本回路为 set 方便异或
        basis = [set(c) for c in basic_cycles_list]

        # 遍历所有基回路的组合 (2^k 种)
        for r in range(1, len(basis) + 1):
            for subset in itertools.combinations(basis, r):
                # 计算组合的对称差
                res = set()
                for s in subset:
                    res = res ^ s # 对称差运算
                if res:
                    cycle_space.append(res)

        # 格式化输出
        output = []
        for c in cycle_space:
            if not c:
                output.append("Φ")
            else:
                sorted_c = sorted(list(c), key=lambda x: int(x[1:]))
                output.append("".join(sorted_c))
        return output

    def visualize(self, tree_adj=None, title="交通网络图"):
        """可视化网络图和生成树"""
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.G, seed=42)

        # 画节点
        nx.draw_networkx_nodes(self.G, pos, node_color='#A0CBE2', node_size=500)
        nx.draw_networkx_labels(self.G, pos)

        # 画所有边
        edge_labels = nx.get_edge_attributes(self.G, 'label')

        if tree_adj is not None:
            # 如果提供了生成树，高亮显示
            all_edges = self.G.edges()
            tree_edges = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if tree_adj[i][j] == 1:
                        tree_edges.append((i, j))

            # 画非树边 (浅色虚线)
            non_tree_edges = [e for e in all_edges if tuple(sorted(e)) not in [tuple(sorted(te)) for te in tree_edges]]
            nx.draw_networkx_edges(self.G, pos, edgelist=non_tree_edges, style='dashed', alpha=0.3)
            # 画树边 (红色加粗)
            nx.draw_networkx_edges(self.G, pos, edgelist=tree_edges, edge_color='red', width=2)
        else:
            nx.draw_networkx_edges(self.G, pos)

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.show()

    @staticmethod
    def from_input():
        """
        支持手动输入和文件输入的接口
        """
        while True:
            try:
                method = int(input("请选择输入邻接矩阵的方式 (0: 手动输入, 1: 来自文件): "))
                if method in [0, 1]: break
            except ValueError: pass

        matrix = []
        if method == 1:
            filename = input("请输入文件名 (如 data.txt): ")
            try:
                with open(filename, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                n = int(lines[0])
                for i in range(1, n + 1):
                    matrix.append(list(map(int, lines[i].split())))
            except Exception as e:
                print(f"读取文件失败: {e}")
                return None
        else:
            n = int(input("请输入路口(节点)数量 N: "))
            print(f"请输入 {n}x{n} 邻接矩阵 (每行 {n} 个数字，空格分隔):")
            for i in range(n):
                row = list(map(int, input(f"第 {i+1} 行: ").split()))
                matrix.append(row)

        return TrafficAnalyzer(matrix)

if __name__ == "__main__":
    analyzer = TrafficAnalyzer.from_input()

    if analyzer:
        print("=交通网络边标定结果 (按行顺序):")
        for u, v, label in analyzer.edge_list:
            print(f"  {label}: ({u}-{v})")

        # 求解生成树个数
        count = analyzer.solve_spanning_trees()
        print(f"\n生成树总个数: {count}")

        # 获取一颗生成树
        t_adj, t_graph, t_labels = analyzer.get_one_spanning_tree()
        print(f"\n其中一颗生成树的相邻矩阵:")
        print(t_adj)
        print(f"   包含边: {t_labels}")

        # 回路系统
        raw_cycles, basic_cycles = analyzer.solve_basic_cycles(t_graph)
        print(f"\n最小绕行单元库 (即基本回路系统):")
        print("   {" + ", ".join(basic_cycles) + "}")

        # 环路空间
        cycle_space = analyzer.solve_cycle_space(raw_cycles)
        print(f"\n完整绕行方案规划 (即环路空间):")
        print("   {" + ", ".join(cycle_space) + "}")

        # 可视化展示
        analyzer.visualize(tree_adj=t_adj, title="交通网络分析 (红色为构建的最小必要路网)")