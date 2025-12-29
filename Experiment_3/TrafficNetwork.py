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

        # 标定边 e1, e2...
        self.edge_list = []
        self.edge_map = {}
        edge_count = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.adj_matrix[i][j] == 1:
                    label = f"e{edge_count}"
                    self.edge_list.append((i, j, label))
                    self.edge_map[tuple(sorted((i, j)))] = label
                    edge_count += 1

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        for u, v, label in self.edge_list:
            self.G.add_edge(u, v, label=label)

    def solve_spanning_trees_count(self):
        """任务3：计算生成树总个数"""
        if not nx.is_connected(self.G): return 0
        D = np.diag(np.sum(self.adj_matrix, axis=1))
        L = D - self.adj_matrix

        return abs(round(np.linalg.det(L[:-1, :-1])))

    def get_all_spanning_trees(self):
        """枚举并返回所有生成树的边集"""
        all_trees_labels = []
        for combo in itertools.combinations(self.edge_list, self.n - 1):
            temp_G = nx.Graph()
            temp_G.add_nodes_from(range(self.n))
            for u, v, label in combo:
                temp_G.add_edge(u, v)
            if nx.is_connected(temp_G):
                labels = sorted([item[2] for item in combo], key=lambda x: int(x[1:]))
                all_trees_labels.append(labels)

        return all_trees_labels

    def get_one_spanning_tree(self):
        """获取一颗用于分析的基准生成树 T1"""
        T = nx.minimum_spanning_tree(self.G)
        T_adj = nx.to_numpy_array(T, weight=None, dtype=int)
        tree_edges = [self.edge_map[tuple(sorted((u, v)))] for u, v in T.edges()]
        return T_adj, T, tree_edges

    def _format_alternating_sequence(self, path_nodes, closing_edge=None):
        """应老师要求，将节点路径转换为 点-边 交替序列"""
        seq = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge_label = self.edge_map[tuple(sorted((u, v)))]
            seq.append(f"v{u}")
            seq.append(edge_label)

        seq.append(f"v{path_nodes[-1]}")
        if closing_edge:
            seq.append(closing_edge)
            seq.append(f"v{path_nodes[0]}")

        return "-".join(seq)

    def solve_basic_cycles(self, T):
        """任务4：求解基本回路系统 (交替序列)"""
        formatted_cycles = []
        raw_edges_list = []
        for u, v, label in self.edge_list:
            if not T.has_edge(u, v):
                path_nodes = nx.shortest_path(T, source=u, target=v)

                # 记录原始边集用于空间计算
                edges = [self.edge_map[tuple(sorted((path_nodes[i], path_nodes[i + 1])))]
                         for i in range(len(path_nodes) - 1)]
                edges.append(label)
                raw_edges_list.append(edges)

                # 记录格式化序列用于显示
                formatted_cycles.append(self._format_alternating_sequence(path_nodes, closing_edge=label))

        return raw_edges_list, formatted_cycles

    def _order_cycle_space_element(self, edge_labels):
        """将环路空间元素格式化为交替序列"""
        if not edge_labels: return "Φ"
        sub_edges = [(u, v) for u, v, lab in self.edge_list if lab in edge_labels]
        sub_G = nx.Graph(sub_edges)

        try:
            cycles = nx.cycle_basis(sub_G)
            if not cycles: return "-".join(sorted(list(edge_labels), key=lambda x: int(x[1:])))

            res = []
            for cyc in cycles:
                path = cyc + [cyc[0]]
                seq = []
                for i in range(len(path) - 1):
                    seq.append(f"v{path[i]}")
                    seq.append(self.edge_map[tuple(sorted((path[i], path[i + 1])))])
                seq.append(f"v{path[0]}")
                res.append("-".join(seq))

            return " ∪ ".join(res)
        except:
            return "-".join(sorted(list(edge_labels), key=lambda x: int(x[1:])))

    def solve_cycle_space(self, raw_basis):
        """任务5：求解环路空间"""
        space_sets = [set()]
        basis = [set(b) for b in raw_basis]

        for r in range(1, len(basis) + 1):
            for subset in itertools.combinations(basis, r):
                res = set()
                for s in subset:
                    res = res ^ s
                if res:
                    space_sets.append(res)

        return [self._order_cycle_space_element(s) for s in space_sets]

    def visualize(self, tree_adj=None, title="交通网络分析"):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw_networkx_nodes(self.G, pos, node_color='#A0CBE2', node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        if tree_adj is not None:
            all_e = list(self.G.edges())
            t_e = [(i, j) for i in range(self.n) for j in range(i + 1, self.n) if tree_adj[i][j] == 1]
            non_t = [e for e in all_e if tuple(sorted(e)) not in [tuple(sorted(te)) for te in t_e]]
            nx.draw_networkx_edges(self.G, pos, edgelist=non_t, style='dashed', alpha=0.3)
            nx.draw_networkx_edges(self.G, pos, edgelist=t_e, edge_color='red', width=2)
        else:
            nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.show()

    @staticmethod
    def from_input():
        while True:
            try:
                method = int(input("请选择输入邻接矩阵的方式 (0: 手动输入, 1: 文件输入): "))
                if method in [0, 1]:
                    break
                print("输入错误，请输入 0 或 1")
            except ValueError:
                print("请输入数字")

        matrix = []

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

        return TrafficAnalyzer(matrix)


if __name__ == "__main__":
    analyzer = TrafficAnalyzer.from_input()

    if analyzer:
        print("边标定结果:")
        for u, v, label in analyzer.edge_list: print(f"   {label}: (v{u}-v{v})")

        # 生成树个数
        count = analyzer.solve_spanning_trees_count()
        print(f"\n生成树总个数: {count}")

        # 显示所有生成树
        all_trees = analyzer.get_all_spanning_trees()
        for i, tree in enumerate(all_trees):
            print(f"T{i + 1}: {tree}")

        # 基准树 T1
        t_adj, t_graph, t_labels = analyzer.get_one_spanning_tree()
        print(f"\n基准生成树 T1 的邻接矩阵:")
        for row in t_adj:
            print(" ".join(map(str, row.astype(int))))

        # 回路系统
        raw_basis, formatted_cycles = analyzer.solve_basic_cycles(t_graph)
        print(f"\n基本回路系统:")
        for i, c in enumerate(formatted_cycles):
            print(f"C{i + 1}: {c}")

        # 环路空间
        cycle_space = analyzer.solve_cycle_space(raw_basis)
        print(f"\n完整环路空间:")
        print("{" + ", \n".join(cycle_space) + "}")

        analyzer.visualize(tree_adj=t_adj, title="交通网络分析 (红色为生成树 T1)")