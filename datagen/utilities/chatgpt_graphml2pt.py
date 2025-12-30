#!/usr/bin/env python3
"""
chatgpt_graphml2pt.py

将目录下的 .graphml 文件批量转换为 PyTorch Geometric 的 Data (.pt) 文件。

输出格式：
    Data(
        x=[N,2],             # 节点特征 (node_type, num_inverted_predecessors)
        edge_index=[2,E],    # 边索引
        edge_attr=[E,1],     # 边属性 (0=BUFF, 1=NOT)
        node_depth=[N]       # 节点深度 (longest path from PI)
    )

兼容 torch-geometric >=2.6
"""

import os
import argparse
import networkx as nx
import torch
from torch_geometric.data import Data
from collections import deque


def compute_node_depths(G, node_list):
    """计算每个节点的逻辑深度"""
    DG = G if G.is_directed() else G.to_directed()

    if nx.is_directed_acyclic_graph(DG):
        indeg = {n: DG.in_degree(n) for n in DG.nodes()}
        depth = {n: 0 for n in DG.nodes()}
        q = deque([n for n, d in indeg.items() if d == 0])
        while q:
            u = q.popleft()
            for v in DG.successors(u):
                depth[v] = max(depth[v], depth[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return [depth[n] for n in node_list]
    else:
        # 如果不是 DAG，就退化为 0
        return [0 for _ in node_list]


def pygDataFromNetworkx(G):
    """将 NetworkX DiGraph 转为 PyG Data"""
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G

    nodes = list(G.nodes())
    node_types = []
    for _, feat_dict in G.nodes(data=True):
        nt = feat_dict.get("node_type", 2)  # 默认 Internal
        node_types.append(int(nt))

    src_idxs, dst_idxs, edge_attrs = [], [], []
    for u, v, feat_dict in G.edges(data=True):
        et = feat_dict.get("edge_type", 0)  # 默认 BUFF
        inv = 1 if int(et) == 1 else 0
        src_idxs.append(u)
        dst_idxs.append(v)
        edge_attrs.append([inv])

    # 每个节点的反相输入数
    inv_count = [0] * len(nodes)
    for s, d, ea in zip(src_idxs, dst_idxs, edge_attrs):
        if ea[0] == 1:
            inv_count[d] += 1

    # 节点特征
    x = torch.tensor([[float(nt), float(ic)] for nt, ic in zip(node_types, inv_count)], dtype=torch.float32)
    edge_index = torch.tensor([src_idxs, dst_idxs], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    # 节点深度
    node_depths = compute_node_depths(G, nodes)
    node_depth = torch.tensor(node_depths, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_depth=node_depth)
    data.num_nodes = len(nodes)
    return data


def process_graphml_file(path, out_dir, verbose=False):
    """单个 graphml → .pt"""
    G = nx.read_graphml(path)
    data = pygDataFromNetworkx(G)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, base + ".pt")
    torch.save(data, out_path)

    if verbose:
        print(f"{base}: Data(x={tuple(data.x.shape)}, "
              f"edge_index={tuple(data.edge_index.shape)}, "
              f"edge_attr={tuple(data.edge_attr.shape)}, "
              f"node_depth={tuple(data.node_depth.shape)})")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert GraphML files to PyG .pt format")
    parser.add_argument("--graphml-dir", required=True, help="输入 .graphml 文件目录")
    parser.add_argument("--out-dir", required=True, help="输出 .pt 文件目录")
    parser.add_argument("--verbose", action="store_true", help="是否打印每个图的形状")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = [f for f in os.listdir(args.graphml_dir) if f.endswith(".graphml")]
    files.sort()

    for f in files:
        fpath = os.path.join(args.graphml_dir, f)
        try:
            process_graphml_file(fpath, args.out_dir, verbose=args.verbose)
        except Exception as e:
            print(f"[ERR] Failed {f}: {e}")

    print(f"✅ Done. Converted {len(files)} graphml files → {args.out_dir}")


if __name__ == "__main__":
    main()

#python chatgpt_graphml2pt.py --graphml-dir data_files/graphml --out-dir data_files/results_pt --verbose