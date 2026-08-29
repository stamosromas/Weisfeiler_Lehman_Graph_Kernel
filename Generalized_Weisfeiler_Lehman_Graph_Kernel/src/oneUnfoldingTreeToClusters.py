"""
Clusters depth-1 unfolding trees into k clusters via Wasserstein k-means on
a [root one-hot | children-label histogram] feature vector.
"""

from typing import Dict, List

import networkx as nx
import numpy as np

from wassersteinKMeans import ModifiedWassersteinKMeans

MAX_CLUSTERING_ATTEMPTS = 25


class compute1UnfTreeToClusterMap:
    def __init__(self, unfoldingTrees: List[nx.DiGraph], maxLabel: int, d: int, k: int, Mr: int, Mc: int):
        self.unfTrees = self._sort_lexicographically(unfoldingTrees)
        self.maxLabel = maxLabel
        self.d = d
        self.treeToVecMapping: Dict[str, List[float]] = {}
        self.k = k
        self.X: List[List[float]] = []
        self.res: Dict[str, int] = {}
        self.main()

    def _compute_root_vec(self, T: nx.DiGraph, max_label: int) -> List[float]:
        vec = [0] * (max_label + 1)
        root = next(n for n, d in T.in_degree() if d == 0)
        vec[T.nodes[root]["label"] - 1] = 1
        return vec

    def _encode_tree_to_string(self, T: nx.DiGraph) -> str:
        all_nodes = [n for n, _ in T.in_degree()]
        root, non_root = all_nodes[0], all_nodes[1:]
        labels = [str(T.nodes[root]["label"])] + [str(T.nodes[n]["label"]) for n in non_root]
        return ".".join(labels)

    def _sort_lexicographically(self, tree_list: List[nx.DiGraph]) -> List[nx.DiGraph]:
        return sorted(tree_list, key=self._encode_tree_to_string)

    def _compute_children_root_subtree_vec(self, T: nx.DiGraph, max_label: int, d: int) -> List[float]:
        vec = [0] * (max_label + 1)
        all_nodes = [n for n, _ in T.in_degree()]
        root, non_root = all_nodes[0], all_nodes[1:]
        vec[-1] = 2 * d - T.out_degree(root)
        for node in non_root:
            vec[T.nodes[node]["label"] - 1] += 1
        return vec

    def tree_label_encoding(self, T: nx.DiGraph) -> str:
        if len(T.nodes()) == 1:
            only_node = next(iter(T.nodes()))
            return f"(1,{T.nodes[only_node]['label']}) "
        parts = []
        for node in T.nodes():
            if T.out_degree(node) != 0:
                children = list(T.successors(node))
                child_str = "#".join(f"({ch},{T.nodes[ch]['label']})" for ch in children)
                parts.append(f"({node},{T.nodes[node]['label']})-->{child_str}")
        return ".".join(parts)

    @staticmethod
    def _map_tree_to_cluster(tree_to_vec: Dict[str, List[float]], cluster_to_vecs: Dict) -> Dict[str, int]:
        tree_to_cluster = {}
        for tree_enc, vec in tree_to_vec.items():
            for cluster, vecs in cluster_to_vecs.items():
                if vec in vecs:
                    tree_to_cluster[tree_enc] = cluster
                    break
        return tree_to_cluster

    @staticmethod
    def _has_empty_cluster(cluster_dict: Dict) -> bool:
        return any(len(members) == 0 for members in cluster_dict.values())

    def main(self) -> None:
        for tree in self.unfTrees:
            encoding = self.tree_label_encoding(tree)
            if encoding not in self.treeToVecMapping:
                vec = self._compute_root_vec(tree, self.maxLabel) + self._compute_children_root_subtree_vec(
                    tree, self.maxLabel, self.d
                )
                self.treeToVecMapping[encoding] = vec
                self.X.append(vec)

        Mr = Mc = np.ones((self.maxLabel + 1, self.maxLabel + 1)) - np.identity(self.maxLabel + 1)

        clustering = None
        for _ in range(MAX_CLUSTERING_ATTEMPTS):
            clustering = ModifiedWassersteinKMeans(self.X, Mr, Mc, self.k, 3, len(Mr), len(Mc))
            if not self._has_empty_cluster(clustering.res):
                break
        else:
            raise RuntimeError(
                f"1-unfolding-tree clustering kept producing empty clusters after "
                f"{MAX_CLUSTERING_ATTEMPTS} attempts; try a smaller k."
            )

        self.res = self._map_tree_to_cluster(self.treeToVecMapping, clustering.res)
