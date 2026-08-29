"""
Builds the final feature vectors for the training and test sets:
  node-label histogram  +  depth-1-unfolding-tree cluster histogram
  +  depth-2-unfolding-tree cluster histogram
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from computeNearestTrees import FindNearestTrees
from oneUnfoldingTreeToClusters import compute1UnfTreeToClusterMap
from trainingAndTestSetSplitter import trainingSetTestSetSplitter
from treeEditDistanceWithCosts import SDTedClass
from treeTransformerToIsomorphicForm import canonicalTreeTransformer
from twoUnfoldingTreeToClusters import compute2UnfTreeToClusterMap

Cascade = Tuple[nx.DiGraph, int]


class computeTrainingTestSetVecs:
    def __init__(self, dataset: List[Cascade], clust1: int, clust2: int, perc: float, maxTotalNodes: int):
        split = trainingSetTestSetSplitter(dataset, perc, maxTotalNodes)
        self.trainingset = split.trainingSet
        self.testset = split.testSet

        self.maxLabel = self._compute_max_label()
        self.d = self._compute_max_degree_in_training_set()
        self.CostMatrix = np.ones((self.maxLabel + 1, self.maxLabel + 1)) - np.identity(self.maxLabel + 1)

        self.k1 = clust1
        self.k2 = clust2

        self.trainingSetVecs: List[List[float]] = []
        self.testSetVecs: List[List[float]] = []

        self.oneUnfTrees: List[nx.DiGraph] = []
        self.twoUnfTrees: List[nx.DiGraph] = []

        self.unf1TreeToClustMap: Dict[str, int] = {}
        self.unf2TreeToClustMap: Dict[str, int] = {}

        self.trainingSetDataSet: List[Tuple[List[float], int]] = []
        self.testSetDataSet: List[Tuple[List[float], int]] = []

        self.main()

    # ---------- dataset-wide statistics ----------

    def _compute_max_label(self) -> int:
        return max(
            max(tree.nodes[node]["label"] for node in tree.nodes()) for tree, _ in self.trainingset
        )

    def _compute_max_degree_in_training_set(self) -> int:
        return max(max(tree.degree(node) for node in tree.nodes()) for tree, _ in self.trainingset)

    # ---------- node-count feature vector ----------

    @staticmethod
    def _compute_node_vec_for_tree(T: nx.DiGraph, max_label: int) -> List[int]:
        vec = [0] * max_label
        for node in T.nodes():
            vec[T.nodes[node]["label"] - 1] += 1
        return vec

    def _compute_training_set_node_vecs(self) -> None:
        self.trainingSetVecs = [
            self._compute_node_vec_for_tree(tree, self.maxLabel) for tree, _ in self.trainingset
        ]

    def _compute_test_set_node_vecs(self) -> None:
        self.testSetVecs = [
            self._compute_node_vec_for_tree(tree, self.maxLabel) for tree, _ in self.testset
        ]

    # ---------- unfolding trees ----------

    @staticmethod
    def _get_node_neighbors(G: nx.DiGraph) -> Dict:
        return {node: list(G.neighbors(node)) for node in G.nodes()}

    @staticmethod
    def _get_leaves(G: nx.DiGraph) -> List:
        return [node for node in G.nodes() if G.out_degree(node) == 0]

    def unfolding_tree(self, G: nx.DiGraph, root, depth: int) -> nx.DiGraph:
        """Breadth-first unfold G starting at `root`, `depth` levels deep,
        into a fresh, re-numbered tree."""
        res = nx.DiGraph()
        res.add_node(1, label=G.nodes[root]["label"])
        node_neighbors = self._get_node_neighbors(G)
        next_id = 2
        old_to_new = {1: root}

        for _ in range(depth):
            for leaf in self._get_leaves(res):
                for neighbor in node_neighbors[old_to_new[leaf]]:
                    res.add_node(next_id, label=G.nodes[neighbor]["label"])
                    res.add_edge(leaf, next_id)
                    old_to_new[next_id] = neighbor
                    next_id += 1
        return res

    @staticmethod
    def tree_label_encoding(T: nx.DiGraph) -> str:
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

    def _canonical_unfolding_tree(self, G: nx.DiGraph, node, depth: int) -> nx.DiGraph:
        tree = self.unfolding_tree(G, node, depth)
        return canonicalTreeTransformer(tree).T

    def _compute_all_unf_trees(self, depth: int) -> List[nx.DiGraph]:
        result: List[nx.DiGraph] = []
        seen_encodings: set = set()
        for tree, _ in self.trainingset:
            for node in tree.nodes():
                canon = self._canonical_unfolding_tree(tree, node, depth)
                encoding = self.tree_label_encoding(canon)
                if encoding not in seen_encodings:
                    seen_encodings.add(encoding)
                    result.append(canon)
        return result

    # ---------- cluster-histogram feature vectors ----------

    def _one_unf_trees_vec(self, T: nx.DiGraph, tree_to_cluster: Dict[str, int]) -> List[float]:
        vec = [0] * len(set(tree_to_cluster.values()))
        for node in T.nodes():
            canon = self._canonical_unfolding_tree(T, node, 1)
            vec[tree_to_cluster[self.tree_label_encoding(canon)]] += 1
        return vec

    def _two_unf_trees_vec(self, T: nx.DiGraph, tree_to_cluster: Dict[str, int]) -> List[float]:
        vec = [0] * len(set(tree_to_cluster.values()))
        for node in T.nodes():
            canon = self._canonical_unfolding_tree(T, node, 2)
            vec[tree_to_cluster[self.tree_label_encoding(canon)]] += 1
        return vec

    def _compute_all_unf_trees_for_graph(self, G: nx.DiGraph, depth: int) -> List[nx.DiGraph]:
        return [self._canonical_unfolding_tree(G, node, depth) for node in G.nodes()]

    def _accumulate_unseen_tree_histogram(
        self,
        unfolding_trees: List[nx.DiGraph],
        cluster_map: Dict[str, int],
        reference_trees: List[nx.DiGraph],
    ) -> List[float]:
        vec = [0.0] * len(set(cluster_map.values()))
        for tree in unfolding_trees:
            encoding = self.tree_label_encoding(tree)
            if encoding in cluster_map:
                vec[cluster_map[encoding]] += 1
                continue
            nearest_trees = FindNearestTrees(reference_trees, tree, self.CostMatrix).res
            for nearest_tree in nearest_trees:
                vec[cluster_map[self.tree_label_encoding(nearest_tree)]] += 1 / len(nearest_trees)
        return vec

    def _compute_test_set_dataset(self) -> None:
        self._compute_test_set_node_vecs()

        for i, (tree, target) in enumerate(self.testset):
            print(f"Building test-set vector {i}")

            one_unf = self._compute_all_unf_trees_for_graph(tree, 1)
            one_hist = self._accumulate_unseen_tree_histogram(
                one_unf, self.unf1TreeToClustMap, self.oneUnfTrees
            )

            two_unf = self._compute_all_unf_trees_for_graph(tree, 2)
            two_hist = self._accumulate_unseen_tree_histogram(
                two_unf, self.unf2TreeToClustMap, self.twoUnfTrees
            )

            full_vec = list(self.testSetVecs[i]) + one_hist + two_hist
            self.testSetVecs[i] = full_vec
            self.testSetDataSet.append((full_vec, target))
            print(f"Finished test-set vector {i}\n{'-' * 55}")

    # ---------- orchestration ----------

    def main(self) -> None:
        print("Computing training-set node vectors...")
        self._compute_training_set_node_vecs()

        print("Clustering 1-unfolding trees...")
        self.oneUnfTrees = self._compute_all_unf_trees(1)
        self.unf1TreeToClustMap = compute1UnfTreeToClusterMap(
            self.oneUnfTrees, self.maxLabel, self.d, self.k1, 3, 5
        ).res

        print("Clustering 2-unfolding trees...")
        self.twoUnfTrees = self._compute_all_unf_trees(2)
        self.unf2TreeToClustMap = compute2UnfTreeToClusterMap(
            self.twoUnfTrees, self.maxLabel, self.d, self.k2
        ).res

        for i, (tree, target) in enumerate(self.trainingset):
            full_vec = (
                list(self.trainingSetVecs[i])
                + self._one_unf_trees_vec(tree, self.unf1TreeToClustMap)
                + self._two_unf_trees_vec(tree, self.unf2TreeToClustMap)
            )
            self.trainingSetVecs[i] = full_vec
            self.trainingSetDataSet.append((full_vec, target))
            print(f"Added training example {i} to the dataset")

        self._compute_test_set_dataset()
