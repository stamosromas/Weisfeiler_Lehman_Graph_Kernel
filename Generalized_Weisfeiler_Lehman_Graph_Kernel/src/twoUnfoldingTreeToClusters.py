"""
Clusters depth-2 unfolding trees into k clusters via Wasserstein k-means,
using tree-edit-distance between the non-isomorphic depth-1 subtrees as the
inner "ground cost" for the children-histogram half of each feature vector.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from treeEditDistanceWithCosts import SDTedClass
from wassersteinKMeans import ModifiedWassersteinKMeans

MAX_CLUSTERING_ATTEMPTS = 25


class compute2UnfTreeToClusterMap:
    def __init__(self, unfTreesList: List[nx.DiGraph], maxLabel: int, d: int, clusts: int):
        self.unfTrees = unfTreesList
        self.maxLabel = maxLabel
        self.CostMat = np.ones((self.maxLabel + 1, self.maxLabel + 1)) - np.identity(self.maxLabel + 1)
        self.d = d
        self.k = clusts
        self.vecs: List[np.ndarray] = []
        self.undTreeEncodingToVecMap: Dict[str, np.ndarray] = {}
        self.treeEncodedToClusterMap: Dict[str, int] = {}
        self.res: Dict[str, int] = {}
        self.main()

    def _compute_root_vec(self, T: nx.DiGraph, max_label: int) -> List[float]:
        vec = [0] * (max_label + 1)
        root = next(n for n, d in T.in_degree() if d == 0)
        vec[T.nodes[root]["label"] - 1] = 1
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

    def _root_children_subtrees(self, T: nx.DiGraph) -> List[nx.DiGraph]:
        root = next(n for n, d in T.in_degree() if d == 0)
        subtrees = []
        for child in T.successors(root):
            subtree = nx.bfs_tree(T, child)
            remap = {old: i + 1 for i, old in enumerate(subtree.nodes())}
            new_subtree = nx.DiGraph()
            for old, new in remap.items():
                new_subtree.add_node(new, label=T.nodes[old]["label"])
            for u, v in subtree.edges():
                new_subtree.add_edge(remap[u], remap[v])
            subtrees.append(new_subtree)
        return subtrees

    def _encode_tree_to_string(self, T: nx.DiGraph) -> str:
        all_nodes = [n for n, _ in T.in_degree()]
        root, non_root = all_nodes[0], all_nodes[1:]
        labels = [str(T.nodes[root]["label"])] + [str(T.nodes[n]["label"]) for n in non_root]
        return ".".join(labels)

    def _sort_lexicographically(self, tree_list: List[nx.DiGraph]) -> List[nx.DiGraph]:
        return sorted(tree_list, key=self._encode_tree_to_string)

    def _compute_subtrees_and_index_map(self) -> Tuple[List[nx.DiGraph], Dict[str, int]]:
        non_iso_trees: List[nx.DiGraph] = []
        seen_encodings: set = set()
        for tree in self.unfTrees:
            for subtree in self._root_children_subtrees(tree):
                enc = self.tree_label_encoding(subtree)
                if enc not in seen_encodings:
                    seen_encodings.add(enc)
                    non_iso_trees.append(subtree)
        non_iso_trees = self._sort_lexicographically(non_iso_trees)
        index_map = {self.tree_label_encoding(t): i for i, t in enumerate(non_iso_trees)}
        return non_iso_trees, index_map

    def _compute_children_subtrees_vector(self, T: nx.DiGraph, subtree_index_map: Dict[str, int], d: int) -> List[float]:
        vec = [0] * (len(subtree_index_map) + 1)
        root = next(n for n, deg in T.in_degree() if deg == 0)
        vec[-1] = 2 * d - T.out_degree(root)
        for subtree in self._root_children_subtrees(T):
            vec[subtree_index_map[self.tree_label_encoding(subtree)]] += 1
        return vec

    def _tree_edit_distance(self, T1: nx.DiGraph, T2: nx.DiGraph) -> float:
        return SDTedClass(T1, T2, self.CostMat).res

    def _compute_subtree_cost_matrix(self, non_iso_trees: List[nx.DiGraph]) -> np.ndarray:
        n = len(non_iso_trees)
        Mc = np.zeros((n + 1, n + 1))
        for i in range(1, n):
            for j in range(i):
                dist = self._tree_edit_distance(non_iso_trees[i], non_iso_trees[j])
                Mc[i][j] = Mc[j][i] = dist
        for i in range(n):
            size = len(non_iso_trees[i].nodes())
            Mc[i][n] = Mc[n][i] = size
        return Mc

    @staticmethod
    def _to_np_arrays(list_of_lists: List[List[float]]) -> List[np.ndarray]:
        return [np.array(elem) for elem in list_of_lists]

    @staticmethod
    def _find_cluster_of_each_tree(tree_to_vec: Dict[str, np.ndarray], cluster_to_vecs: Dict) -> Dict[str, int]:
        tree_to_cluster = {}
        for tree_enc, vec in tree_to_vec.items():
            for cluster, vecs in cluster_to_vecs.items():
                if any(np.array_equal(vec, v) for v in vecs):
                    tree_to_cluster[tree_enc] = cluster
        return tree_to_cluster

    @staticmethod
    def _cluster_dict_to_python_lists(cluster_dict: Dict) -> Dict:
        return {
            cluster: [v.tolist() if isinstance(v, np.ndarray) else v for v in vecs]
            for cluster, vecs in cluster_dict.items()
        }

    @staticmethod
    def _has_empty_cluster(cluster_dict: Dict) -> bool:
        return any(len(members) == 0 for members in cluster_dict.values())

    def main(self) -> None:
        non_iso_subtrees, subtree_index_map = self._compute_subtrees_and_index_map()
        for tree in self.unfTrees:
            vec = self._compute_root_vec(tree, self.maxLabel) + self._compute_children_subtrees_vector(
                tree, subtree_index_map, self.d
            )
            self.vecs.append(vec)
            self.undTreeEncodingToVecMap[self.tree_label_encoding(tree)] = vec

        self.vecs = self._to_np_arrays(self.vecs)
        Mr = np.ones((self.maxLabel + 1, self.maxLabel + 1)) - np.identity(self.maxLabel + 1)
        Mc = self._compute_subtree_cost_matrix(non_iso_subtrees)

        clustering = None
        for _ in range(MAX_CLUSTERING_ATTEMPTS):
            clustering = ModifiedWassersteinKMeans(
                self.vecs, Mr, Mc, clusts=self.k, numOfIters=5, n_r=len(Mr), n_c=len(Mc)
            )
            if not self._has_empty_cluster(clustering.res):
                break
        else:
            raise RuntimeError(
                f"2-unfolding-tree clustering kept producing empty clusters after "
                f"{MAX_CLUSTERING_ATTEMPTS} attempts; try a smaller k."
            )

        cluster_to_vecs = self._cluster_dict_to_python_lists(clustering.res)
        self.treeEncodedToClusterMap = self.res = self._find_cluster_of_each_tree(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.undTreeEncodingToVecMap.items()},
            cluster_to_vecs,
        )
