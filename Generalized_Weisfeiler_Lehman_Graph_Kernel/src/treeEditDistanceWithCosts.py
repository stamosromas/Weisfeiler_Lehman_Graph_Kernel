from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


class SDTedClass:
    """Ordered tree edit distance with an explicit cost matrix.

    Parameters
    ----------
    T1, T2 : nx.DiGraph
        Rooted trees; every node must have an integer 'label' attribute in
        [1, CostMatrix.shape[0]].
    CostMatrix : np.ndarray
        Square cost matrix. CostMatrix[i-1][j-1] is the cost of relabeling a
        node with label i as label j. The LAST row/column is treated as the
        cost of matching a real label against a virtual "null" label, i.e.
        the cost of deleting/inserting a node with that label. Because of
        this, CostMatrix must be sized (max_label + 1) x (max_label + 1) —
        NOT max_label x max_label — everywhere it is constructed.
    cache : dict, optional
        Internal memoization cache, keyed by (encoding(T1), encoding(T2)).
        Leave as None for a top-level call; recursive calls pass the same
        dict down automatically.
    """

    def __init__(
        self,
        T1: nx.DiGraph,
        T2: nx.DiGraph,
        CostMatrix: np.ndarray,
        cache: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.T1 = T1
        self.T2 = T2
        self.Sigma = CostMatrix
        self.cache: Dict[Tuple[str, str], float] = cache if cache is not None else {}
        self.res = 0.0
        self.FrT1: List[nx.DiGraph] = []
        self.FrT2: List[nx.DiGraph] = []
        self.main()


    def root_children_subtrees(self, T: nx.DiGraph) -> List[nx.DiGraph]:
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

    def compute_subtree_forest(self, T: nx.DiGraph, empty_graphs: int) -> List[nx.DiGraph]:
        forest = self.root_children_subtrees(T)
        forest.extend(nx.DiGraph() for _ in range(empty_graphs))
        return forest

    def cost_of_deleting_all_nodes(self, T: nx.DiGraph) -> float:
        return sum(self.Sigma[T.nodes[node]["label"] - 1][-1] for node in T.nodes())

    def min_cost_bipartite_matching(self, M: List[List[float]]) -> float:
        M = np.array(M)
        row_ind, col_ind = linear_sum_assignment(M)
        return M[row_ind, col_ind].sum()

    def root_relabel_cost(self, label1: int, label2: int) -> float:
        return self.Sigma[label1 - 1][label2 - 1]

    def subtree_distance(self, Ti: nx.DiGraph, Tj: nx.DiGraph) -> float:
        return SDTedClass(Ti, Tj, self.Sigma, cache=self.cache).res

    def forest_element_distance(self, Ti: nx.DiGraph, Tj: nx.DiGraph) -> float:
        ti_is_real = Ti in self.FrT1
        tj_is_real = Tj in self.FrT2
        if ti_is_real and tj_is_real:
            return self.subtree_distance(Ti, Tj)
        if ti_is_real:
            return self.cost_of_deleting_all_nodes(Tj)
        if tj_is_real:
            return self.cost_of_deleting_all_nodes(Ti)
        return 0.0

    @staticmethod
    def node_children(T: nx.DiGraph, node) -> List:
        return list(T.successors(node))

    def tree_label_encoding(self, T: nx.DiGraph) -> str:
        if len(T.nodes()) == 1:
            only_node = next(iter(T.nodes()))
            return f"(1,{T.nodes[only_node]['label']}) "
        parts = []
        for node in T.nodes():
            if T.out_degree(node) != 0:
                children = self.node_children(T, node)
                child_str = "#".join(f"({ch},{T.nodes[ch]['label']})" for ch in children)
                parts.append(f"({node},{T.nodes[node]['label']})-->{child_str}")
        return ".".join(parts)


    def main(self) -> None:
        if len(self.T1.nodes()) == 1 and len(self.T2.nodes()) == 1:
            (n1,) = self.T1.nodes()
            (n2,) = self.T2.nodes()
            label1 = self.T1.nodes[n1]["label"]
            label2 = self.T2.nodes[n2]["label"]
            self.res = self.root_relabel_cost(label1, label2)
            key1 = (self.tree_label_encoding(self.T1), self.tree_label_encoding(self.T2))
            self.cache[key1] = self.res
            self.cache[(key1[1], key1[0])] = self.res
            return

        root1 = min(n for n, d in self.T1.in_degree() if d == 0)
        root2 = min(n for n, d in self.T2.in_degree() if d == 0)
        root_deg1 = self.T1.out_degree(root1)
        root_deg2 = self.T2.out_degree(root2)

        self.FrT1 = self.compute_subtree_forest(self.T1, root_deg2)
        self.FrT2 = self.compute_subtree_forest(self.T2, root_deg1)

        M = [[0.0] * len(self.FrT2) for _ in range(len(self.FrT1))]
        for i, ti in enumerate(self.FrT1):
            for j, tj in enumerate(self.FrT2):
                if len(ti.nodes()) == 0:
                    M[i][j] = self.cost_of_deleting_all_nodes(tj)
                    continue
                if len(tj.nodes()) == 0:
                    M[i][j] = self.cost_of_deleting_all_nodes(ti)
                    continue
                key = (self.tree_label_encoding(ti), self.tree_label_encoding(tj))
                if key in self.cache:
                    M[i][j] = self.cache[key]
                else:
                    dist = self.forest_element_distance(ti, tj)
                    M[i][j] = dist
                    self.cache[key] = dist
                    self.cache[(key[1], key[0])] = dist

        min_cost_matching = self.min_cost_bipartite_matching(M)
        self.res = (
            self.root_relabel_cost(self.T1.nodes[root1]["label"], self.T2.nodes[root2]["label"])
            + min_cost_matching
        )
