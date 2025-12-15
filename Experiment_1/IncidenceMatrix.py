import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Matrix:
    def __init__(self, _row_: int, _col_: int, _matrix_, _type_: int):
        self._row_ = _row_
        self._col_ = _col_
        self._type_ = _type_
        """
        type = 0 : 相邻矩阵
        type = 1 : 关联矩阵
        """

        if isinstance(_matrix_, np.ndarray):
            self._matrix_ = _matrix_
        else:
            self._matrix_ = np.array(_matrix_, dtype=int)

    # 生成相邻矩阵
    def get_adj_matrix(self):
        if self._type_ == 0:
            return self

        elif self._type_ == 1:
            _adj_matrix = np.zeros((self._row_, self._row_), dtype=int)

            # 遍历列
            for j in range(self._col_):
                col_data = self._matrix_[:, j]
                nodes = np.where(col_data == 1)[0]
                # 处理普通边
                if len(nodes) == 2:
                    u, v = nodes[0], nodes[1]
                    _adj_matrix[u][v] += 1
                    _adj_matrix[v][u] += 1
                # 处理自环边
                elif len(nodes) == 1:
                    u = nodes[0]
                    _adj_matrix[u][u] = 1

            return Matrix(self._row_, self._row_, _adj_matrix, 0)

        # 未知的矩阵类型
        return None

    # 打印矩阵
    def show_matrix(self):
        if self._type_ == 0:
            print("     ", end='')
            for i in range(self._col_):
                print("v" + str(i + 1), end=' ')
            print()
        if self._type_ == 1:
            print("     ", end='')
            for i in range(self._col_):
                print("e" + str(i + 1), end=' ')
            print()

        for i in range(self._row_):
            print("v" + str(i + 1) + " [  ", end='')
            for j in range(self._col_):
                print(self._matrix_[i][j], end='  ')
            print("]")


    def draw_matrix_graph(self):
        plt.figure(figsize=(10, 8))
        G = nx.MultiGraph()

        # 节点 v1, v2...
        node_labels = {}
        for i in range(self._row_):
            G.add_node(i)
            node_labels[i] = f"$v_{{{i+1}}}$"

        edge_labels = {}

        if self._type_ == 1:
            for j in range(self._col_):
                col_data = self._matrix_[:, j]
                nodes = np.where(col_data == 1)[0]

                edge_name = f"$e_{{{j+1}}}$" # 边 e_1, e_2...

                if len(nodes) == 2:
                    u, v = nodes[0], nodes[1]
                    G.add_edge(u, v, key=j, label=edge_name)
                    edge_labels[(u, v)] = edge_name

                elif len(nodes) == 1: # 自环
                    u = nodes[0]
                    G.add_edge(u, u, key=j, label=edge_name)
                    edge_labels[(u, u)] = edge_name

        else:
            _adj = self._matrix_

            rows, cols = _adj.shape
            e_count = 1
            for r in range(rows):
                for c in range(r, cols): # 只遍历上三角
                    count = _adj[r][c]
                    if count > 0:
                        for _ in range(count):
                            edge_name = f"e{e_count}"
                            G.add_edge(r, c, label=edge_name)
                            e_count += 1

        pos = nx.spring_layout(G, seed=42, k=0.8)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#A0CBE2', edgecolors='black')

        # 绘制点标签 (v1, v2...)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, font_family='sans-serif')

        # 绘制多重边
        ax = plt.gca()
        for u, v, key, data in G.edges(keys=True, data=True):
            rad = 0.1 * (key % 3 + 1) * (1 if key % 2 == 0 else -1)

            if u == v:
                rad = 0.4

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=1.5,
                alpha=0.6,
                edge_color='black',
                connectionstyle=f'arc3, rad={rad}'
            )

            # 绘制边标签 (e1, e2...)
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            lbl_x = (x1 + x2) / 2
            lbl_y = (y1 + y2) / 2

            offset_x = (y1 - y2) * rad * 0.5
            offset_y = (x2 - x1) * rad * 0.5

            # 自环的标签特殊处理
            if u == v:
                lbl_y += 0.15
            else:
                lbl_x += offset_x
                lbl_y += offset_y

            plt.text(lbl_x, lbl_y, data['label'],
                     size=10, color='red',
                     horizontalalignment='center',
                     verticalalignment='center',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.axis('off')
        plt.show()

    # 分析矩阵
    def analyse_matrix(self):
        print("开始分析...")
        _adj_matrix = self.get_adj_matrix()
        if _adj_matrix is None:
            return

        A = _adj_matrix._matrix_
        n = self._row_ # 顶点数

        # 任务三 判断每个顶点的度数、邻域
        for i in range(n):
            # 计算度数
            # 无向图度数 = 行和(如果有自环还要再加一次)
            degree = np.sum(A[i]) + A[i][i]

            # 计算邻域
            # 找到该行大于0的列索引
            neighbor_indices = np.where(A[i] > 0)[0]
            neighbor_names = [f"v{idx + 1}" for idx in neighbor_indices]

            print(f"v{i + 1} 度数: {degree} 邻域: {{{', '.join(neighbor_names)}}}")

        # 任务四 判断图是否为简单图
        # 条件1：无自环
        has_loops = np.any(np.diagonal(A) > 0)
        # 条件2：无多重边
        has_parallel_edges = np.any(A > 1)

        if not has_loops and not has_parallel_edges:
            print("该图是简单图")
        else:
            print("该图不是简单图")
            if has_loops:
                print(f"存在自环 (顶点 {[f'v{i+1}' for i in np.where(np.diagonal(A)>0)[0]]})")
            if has_parallel_edges:
                print("存在多重边 (两个顶点之间有多条边)")

        # 任务五 判断无向图的连通分支数及分支包含的顶点
        G_analysis = nx.from_numpy_array(A)

        if nx.is_connected(G_analysis):
            print("连通性：连通")
            print("连通分支数：1")
            print(f"包含所有顶点")
        else:
            components = list(nx.connected_components(G_analysis))
            print("连通性：不连通")
            print(f"连通分支数：{len(components)}")

            for idx, comp in enumerate(components):
                comp_nodes = sorted(list(comp))
                comp_names = [f"v{node + 1}" for node in comp_nodes]
                print(f"分支 {idx + 1}: {{{', '.join(comp_names)}}}")

    @staticmethod
    def from_input():
        method = int(input("请选择输入矩阵的方式 (0: 手动输入, 1: 文件输入): "))
        if method == 1:
            filename = input("请输入文件名: ")
            with open(filename, 'r') as f:
                lines = f.readlines()

            row, col = map(int, lines[0].strip().split()) # 读取行列数
            matrix = []
            for i in range(1, row + 1):
                row_elements = list(map(int, lines[i].strip().split())) # 逐行读取矩阵
                matrix.append(row_elements)

            matrix_type = int(lines[row + 1].strip()) # 矩阵类型
            return Matrix(row, col, matrix, matrix_type)
        else:
            row = int(input("请输入矩阵的行数: "))
            col = int(input("请输入矩阵的列数: "))

            matrix = []
            print(f"请输入矩阵的元素 (共 {row} 行，每行 {col} 个数字):")
            for i in range(row):
                while True:
                    line = input(f"请输入第 {i + 1} 行元素: ")
                    row_elements = list(map(int, line.split()))
                    if len(row_elements) != col:
                        print(f"输入错误：该行应当有 {col} 个数字，请重新输入")
                    else:
                        matrix.append(row_elements)
                        break

            matrix_type = int(input("请输入矩阵类型 (0: 相邻矩阵, 1: 关联矩阵): "))
            return Matrix(row, col, matrix, matrix_type)

if __name__ == "__main__":
    # 从用户输入获取矩阵
    incidence_matrix = Matrix.from_input()

    # 任务一 绘制图形
    incidence_matrix.draw_matrix_graph()

    # 任务二 打印关联矩阵和相邻矩阵
    adj_matrix = incidence_matrix.get_adj_matrix()

    print("\n关联矩阵:")
    incidence_matrix.show_matrix()

    print("\n相邻矩阵:")
    adj_matrix.show_matrix()

    print("\n分析矩阵信息")
    adj_matrix.analyse_matrix()