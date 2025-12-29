import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import operator

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Polynomial:
    """辅助类：处理色多项式的符号运算"""
    def __init__(self, coeffs=None):
        self.coeffs = defaultdict(int, coeffs) if coeffs else defaultdict(int)

    def __add__(self, other):
        res = self.coeffs.copy()
        for p, c in other.coeffs.items():
            res[p] += c
        return Polynomial(res)

    def __sub__(self, other):
        res = self.coeffs.copy()
        for p, c in other.coeffs.items():
            res[p] -= c
        return Polynomial(res)

    def evaluate(self, k):
        return sum(c * (k ** p) for p, c in self.coeffs.items())

    def __str__(self):
        terms = []
        for p in sorted(self.coeffs.keys(), reverse=True):
            c = self.coeffs[p]
            if c == 0: continue

            # 符号处理
            sign = "+ " if c > 0 else "- "
            if not terms and c > 0: sign = "" # 首项正号省略
            if not terms and c < 0: sign = "-"

            abs_c = abs(c)
            # 系数为1且不是常数项时省略数字
            c_str = str(abs_c) if abs_c != 1 or p == 0 else ""
            p_str = f"k^{p}" if p > 1 else ("k" if p == 1 else "")

            terms.append(f"{sign}{c_str}{p_str}")
        return "".join(terms).strip() if terms else "0"

class FrequencyAllocator:
    def __init__(self, matrix):
        self.adj_matrix = np.array(matrix, dtype=int)
        self.n = len(self.adj_matrix)

        # 初始化 NetworkX 图
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] == 1:
                    self.G.add_edge(i, j)

        self.chromatic_num = None
        self.poly = None

    # --- 核心算法 1: 求点色数 (使用最大度优先优化 + 回溯) ---
    def solve_chromatic_number(self):
        if self.chromatic_num is not None: return self.chromatic_num

        # 优化策略：按节点度数从大到小排序 (Welsh-Powell 思想)
        # 优先着色限制最多的节点，能显著剪枝
        degrees = dict(self.G.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        def is_safe(node, c, assignment):
            for neighbor in self.G.neighbors(node):
                if neighbor in assignment and assignment[neighbor] == c:
                    return False
            return True

        def backtrack(node_idx, k_colors, assignment):
            if node_idx == self.n:
                return True

            node = sorted_nodes[node_idx]
            for c in range(1, k_colors + 1):
                if is_safe(node, c, assignment):
                    assignment[node] = c
                    if backtrack(node_idx + 1, k_colors, assignment):
                        return True
                    del assignment[node]
            return False

        # 从 1 到 n 尝试寻找最小色数
        for k in range(1, self.n + 1):
            if backtrack(0, k, {}):
                self.chromatic_num = k
                return k
        return self.n

    # --- 核心算法 2: 求色多项式 (使用删除-收缩定理) ---
    def _deletion_contraction(self, G_curr):
        # 递归基：无边图 (Null Graph) -> k^n
        if G_curr.number_of_edges() == 0:
            return Polynomial({G_curr.number_of_nodes(): 1})

        # 选择一条边 e=(u,v)
        # 优化：每次选择第一条边即可
        u, v = list(G_curr.edges())[0]

        # 1. 删除 (Deletion): G - e
        G_del = G_curr.copy()
        G_del.remove_edge(u, v)

        # 2. 收缩 (Contraction): G . e
        # NetworkX 的 contracted_nodes 会产生多重边和自环
        # 色多项式计算中，收缩后的图应当处理为简单图（去除自环和重边）
        G_cont = nx.Graph()
        mapping = {node: (u if node == v else node) for node in G_curr.nodes()}

        # 重新构建收缩后的边集
        edges_to_add = set()
        for x, y in G_curr.edges():
            nx_map, ny_map = mapping[x], mapping[y]
            if nx_map != ny_map: # 去除自环
                # 排序元组以去重 (无向图)
                edge = tuple(sorted((nx_map, ny_map)))
                edges_to_add.add(edge)

        # 节点数减少 1 (v 被合并进 u)
        # 注意：这里需要保持原递归逻辑中的节点数量概念
        # 收缩后的图节点数为 n-1
        G_cont.add_nodes_from(set(mapping.values()))
        G_cont.add_edges_from(edges_to_add)

        # P(G) = P(G-e) - P(G.e)
        return self._deletion_contraction(G_del) - self._deletion_contraction(G_cont)

    def get_chromatic_polynomial(self):
        if self.poly is None:
            self.poly = self._deletion_contraction(self.G)
        return self.poly

    def analyze(self, k_list):
        print("\n" + "="*60)
        print(f"图规模: {self.n} 顶点 | 待分析频段数 k = {k_list}")
        print("="*60)

        # 1. 预计算色多项式 (只计算一次)
        poly = self.get_chromatic_polynomial()
        # 2. 预计算点色数
        chi = self.solve_chromatic_number()

        for k in k_list:
            print(f"\n>>> 场景: 当频段数 k = {k}")

            # 问题 1
            can_assign = k >= chi
            print(f">>> 问题 1: 能否在 {k} 个频段内完成分配?")
            print(f"    - 分析图的点色数: {chi}")
            print(f"    - 结论: [{'可以' if can_assign else '不可'}] (因为 {k} {'>=' if can_assign else '<'} {chi})")

            # 问题 2
            ways = poly.evaluate(k)
            print(f">>> 问题 2: 所有可能合法的分配方案数是多少?")
            print(f"    - 色多项式 P(G, k) = {poly}")
            print(f"    - 代入 k={k} 计算...")
            print(f"    - 结论: 共 [{ways}] 种方案")

    def visualize(self):
        plt.figure(figsize=(6, 5))
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(self.G, pos, with_labels=True, node_color='#A0CBE2',
                edge_color='gray', node_size=500, font_size=12)
        plt.title(f"无线基站网络拓扑 (N={self.n})")
        plt.show()

    @staticmethod
    def from_input():
        """
        复用的输入函数，并在末尾增加了读取 k 值的功能
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

                if not lines: raise ValueError("文件为空")
                n = int(lines[0])

                if len(lines) < n + 1:
                    raise ValueError("文件行数不足")

                print(f"检测到 {n} 个节点，正在读取矩阵...")

                for i in range(1, n + 1):
                    # 兼容 float 输入并转为 int
                    row_elements = list(map(int, map(float, lines[i].split())))
                    if len(row_elements) != n:
                        raise ValueError(f"第 {i} 行数据列数不正确，应为 {n} 列")
                    matrix.append(row_elements)

            except FileNotFoundError:
                print(f"错误：找不到文件 {filename}")
                return None, None
            except Exception as e:
                print(f"文件读取错误: {e}")
                return None, None

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
                        row_elements = list(map(int, map(float, line.split())))
                        if len(row_elements) != n:
                            print(f"错误：该行应当有 {n} 个数字，当前有 {len(row_elements)} 个")
                            continue
                        matrix.append(row_elements)
                        break
                    except ValueError:
                        print("输入包含非数字字符，请重试")

        # 读取 k 值
        while True:
            try:
                k_input = input("请输入要测试的频段数 k (多个值用空格分隔，例如 '2 3 4'): ")
                k_vals = list(map(int, k_input.split()))
                if len(k_vals) > 0:
                    break
            except ValueError:
                print("请输入有效的整数列表")

        return FrequencyAllocator(matrix), k_vals


if __name__ == "__main__":
    allocator, k_values = FrequencyAllocator.from_input()

    if allocator:
        allocator.analyze(k_values)
        print("\n正在生成可视化图表...")
        allocator.visualize()