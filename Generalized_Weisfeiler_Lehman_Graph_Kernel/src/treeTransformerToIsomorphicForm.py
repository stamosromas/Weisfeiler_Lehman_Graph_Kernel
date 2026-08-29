"""
Transforms an unfolding tree into a canonical (isomorphism-invariant) form:
pads it to a uniform depth with zero-labeled placeholder nodes, repeatedly
collapses each node's children into a single re-labeled "subtree symbol",
then unfolds that symbol table back out and strips the placeholder nodes.
"""

from typing import Dict, List

import networkx as nx


class canonicalTreeTransformer:
    def __init__(self, T: nx.DiGraph):
        self.T = T
        self.labelSubtreeString: Dict[int, str] = self._create_initial_mapping()
        self.initialMaxDepth = self._compute_max_depth()
        self.main()

    def _create_initial_mapping(self) -> Dict[int, str]:
        labels = {self.T.nodes[node]["label"] for node in self.T.nodes()}
        return {label: str(label) for label in labels}

    def _compute_max_depth(self) -> int:
        root = next(n for n, d in self.T.in_degree() if d == 0)
        leaves = [n for n in self.T.nodes() if self.T.out_degree(n) == 0]
        shortest_paths = dict(nx.all_pairs_shortest_path_length(self.T))
        return max(shortest_paths[root][leaf] for leaf in leaves)

    @staticmethod
    def _add_zero_nodes(T: nx.DiGraph, node, iters: int) -> nx.DiGraph:
        next_id = max(T.nodes()) + 1
        prev_node = node
        for _ in range(iters):
            T.add_node(next_id, label=0)
            T.add_edge(prev_node, next_id)
            prev_node = next_id
            next_id += 1
        return T

    def _pad_with_zero_nodes(self) -> None:
        root = next(n for n, d in self.T.in_degree() if d == 0)
        leaves = [n for n in self.T.nodes() if self.T.out_degree(n) == 0]
        shortest_paths_from_root = dict(nx.all_pairs_shortest_path_length(self.T))[root]
        max_depth = self._compute_max_depth()
        for leaf in leaves:
            if shortest_paths_from_root[leaf] != max_depth:
                self.T = self._add_zero_nodes(
                    self.T, leaf, max_depth - shortest_paths_from_root[leaf]
                )

    def _sort_node_labels(self) -> None:
        root = next(n for n, d in self.T.in_degree() if d == 0)
        shortest_paths_from_root = dict(nx.all_pairs_shortest_path_length(self.T))[root]
        max_depth = self._compute_max_depth()
        leaf_parents = [
            n for n in self.T.nodes() if shortest_paths_from_root[n] == max_depth - 1
        ]
        for node in leaf_parents:
            children = list(self.T.successors(node))
            if len(children) > 1:
                labels = sorted(self.T.nodes[ch]["label"] for ch in children)
                for label, child in zip(labels, children):
                    self.T.nodes[child]["label"] = label

    def _compute_max_label(self) -> int:
        return max(self.T.nodes[node]["label"] for node in self.T.nodes())

    def _create_node_enum(self, subtree_strs: List[str]) -> Dict[str, int]:
        distinct_sorted = sorted(set(subtree_strs))
        next_label = self._compute_max_label() + 1
        mapping = {}
        for st in distinct_sorted:
            mapping[st] = next_label
            next_label += 1
        return mapping

    def _update_label_subtrees(self, subtree_label_map: Dict[str, int]) -> None:
        for subtree_str, label in subtree_label_map.items():
            self.labelSubtreeString[label] = subtree_str

    def _create_subtree_str_enum(self) -> Dict[str, int]:
        root = next(n for n, d in self.T.in_degree() if d == 0)
        shortest_paths_from_root = dict(nx.all_pairs_shortest_path_length(self.T))[root]
        max_depth = self._compute_max_depth()
        leaf_parents = [
            n for n in self.T.nodes() if shortest_paths_from_root[n] == max_depth - 1
        ]
        subtree_strings = []
        for leaf_parent in leaf_parents:
            parts = [str(self.T.nodes[leaf_parent]["label"])]
            parts += [str(self.T.nodes[ch]["label"]) for ch in self.T.successors(leaf_parent)]
            subtree_strings.append(".".join(parts))
        subtree_str_map = self._create_node_enum(subtree_strings)
        self._update_label_subtrees(subtree_str_map)
        return subtree_str_map

    def _compute_subtree_str(self, node) -> str:
        parts = [str(self.T.nodes[node]["label"])]
        parts += [str(self.T.nodes[ch]["label"]) for ch in self.T.successors(node)]
        return ".".join(parts)

    def _node_replacement(self, subtree_map: Dict[str, int]) -> None:
        root = next(n for n, d in self.T.in_degree() if d == 0)
        shortest_paths_from_root = dict(nx.all_pairs_shortest_path_length(self.T))[root]
        max_depth = self._compute_max_depth()
        leaf_parents = [
            n for n in self.T.nodes() if shortest_paths_from_root[n] == max_depth - 1
        ]
        for node in leaf_parents:
            subtree_str = self._compute_subtree_str(node)
            self.T.nodes[node]["label"] = subtree_map[subtree_str]
            for descendant in list(nx.descendants(self.T, node)):
                self.T.remove_node(descendant)

    def _unfold_tree(self) -> None:
        next_id = max(self.T.nodes()) + 1
        for _ in range(self.initialMaxDepth - 1):
            leaves = [n for n in self.T.nodes() if self.T.out_degree(n) == 0]
            for leaf in leaves:
                leaf_label = self.T.nodes[leaf]["label"]
                subtree_nodes = self.labelSubtreeString[leaf_label].split(".")
                for child_label in subtree_nodes[1:]:
                    self.T.add_node(next_id, label=int(child_label))
                    self.T.add_edge(leaf, next_id)
                    next_id += 1
                self.T.nodes[leaf]["label"] = int(subtree_nodes[0])
        for node in list(self.T.nodes()):
            if self.T.nodes[node]["label"] == 0:
                self.T.remove_node(node)

    def _reenumerate_tree_nodes(self) -> None:
        kept_nodes = [node for node in self.T.nodes() if self.T.nodes[node]["label"] != 0]
        kept_node_set = set(kept_nodes)
        kept_edges = [
            (u, v) for u, v in self.T.edges() if u in kept_node_set and v in kept_node_set
        ]

        new_id = {old: i + 1 for i, old in enumerate(kept_nodes)}
        new_attribs = {new_id[old]: self.T.nodes[old]["label"] for old in kept_nodes}
        new_edges = [(new_id[u], new_id[v]) for u, v in kept_edges]

        g_new = nx.DiGraph()
        g_new.add_nodes_from(new_id.values())
        nx.set_node_attributes(g_new, new_attribs, name="label")
        g_new.add_edges_from(new_edges)

        self.T = g_new

    def main(self) -> None:
        self._pad_with_zero_nodes()
        while self._compute_max_depth() > 1:
            self._sort_node_labels()
            mapping = self._create_subtree_str_enum()
            self._node_replacement(mapping)
        self._sort_node_labels()
        self._unfold_tree()
        self._reenumerate_tree_nodes()
