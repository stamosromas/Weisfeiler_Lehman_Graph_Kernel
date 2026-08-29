"""
Labels the nodes of each cascade tree in a dataset according to one of
several strategies (out-degree binning, k-means on a node statistic, user
identity heuristics, etc, etc).
"""

from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np

from kMeansIn1d import kMeans1d

Cascade = Tuple[nx.DiGraph, str]


class nodeLabelingOfDataSet:

    _METHODS: Dict[str, str] = {
        "cascade-based-log-bin": "_node_labeling_cascades_log_bin",
        "graph-based-out-deg-sum": "_log_bin_out_deg_sum",
        "graph-based-median-out-deg": "_log_bin_mean_out_deg",
        "graph-based-mean-out-deg": "_mean_out_deg_dataset_labeling",
        "good-bad-user-freq": "_good_bad_users_freq_participation",
        "cascade-based-k-means": "_cascade_based_kmeans",
        "graph-based-k-means": "_graph_based_kmeans",
        "out-deg-statistics": "_out_deg_stats_tree_labeling",
        "cascade-based-max-depth": "_max_depth_dataset_labeling",
        "cascade-based-kmeans-tot-descendants": "_cascade_based_descendant_kmeans_ds",
        "cascade-based-tot-leaves-kmeans": "_tot_leaves_kmeans",
        "cascade-based-max-leaf-root-depth": "_root_deepest_leaf_node_labeling_dataset",
        "good-heuristic-using-user-identities": "_create_heuristic_dataset",
        "all-nodes-same-label": "_set_label_one_to_all_nodes",
    }

    def __init__(self, dataSet: List[Cascade], method: str):
        handler_name = self._METHODS.get(method, "_mean_out_deg_per_user_dataset")
        self.dataSet = getattr(self, handler_name)(dataSet)

    @staticmethod
    def target_to_int(target: str) -> int:
        return 1 if target == "true" else 0

    # ---------- simple label assignment ----------

    def _set_label_one_to_all_nodes(self, dataset: List[Cascade]) -> List[Cascade]:
        new_dataset = []
        for tree, target in dataset:
            g = nx.DiGraph()
            for node in tree.nodes():
                g.add_node(node, label=1)
            g.add_edges_from(tree.edges())
            new_dataset.append((g, self.target_to_int(target)))
        return new_dataset

    def _mean_out_deg_per_user_dataset(self, dataset: List[Cascade]) -> List[Cascade]:
        return self._mean_out_deg_dataset_labeling(dataset)

    # ---------- clustering-based user classification ----------

    @staticmethod
    def _classify_to_cluster(vals: List[float]) -> int:
        if all(v == 0 for v in vals):
            return 1 if len(vals) > 2 else 2
        mean_val = sum(vals) / len(vals)
        if mean_val < 0.3:
            return 2
        if mean_val < 1:
            return 3
        return 4

    @staticmethod
    def _label_tree_from_dict(T: nx.DiGraph, label_map: Dict) -> nx.DiGraph:
        res = nx.DiGraph()
        for node in T.nodes():
            res.add_node(node, label=label_map[T.nodes[node]["label"]])
        res.add_edges_from(T.edges())
        return res

    def _create_heuristic_dataset(self, dataset: List[Cascade]) -> List[Cascade]:
        users_out_deg: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                users_out_deg.setdefault(user, []).append(tree.out_degree(node))

        user_to_cluster = {
            user: self._classify_to_cluster(degs) for user, degs in users_out_deg.items()
        }

        new_dataset = []
        for tree, target in dataset:
            new_dataset.append(
                (self._label_tree_from_dict(tree, user_to_cluster), self.target_to_int(target))
            )
        return new_dataset

    # ---------- depth-based labeling ----------

    @staticmethod
    def _max_dict_value(d: Dict) -> int:
        return max(d.values(), default=0)

    def _root_deepest_leaf_node_labeling(self, T: nx.DiGraph) -> nx.DiGraph:
        root = next(n for n, d in T.in_degree() if d == 0)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(T))
        dist_from_root = shortest_paths[root]

        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        res.add_edges_from(T.edges())

        new_attribs = {
            node: abs(self._max_dict_value(shortest_paths[node]) - dist_from_root[node]) + 1
            for node in T.nodes()
        }
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _root_deepest_leaf_node_labeling_dataset(self, dataset: List[Cascade]) -> List[Cascade]:
        return [
            (self._root_deepest_leaf_node_labeling(tree), target) for tree, target in dataset
        ]

    def _max_depth_labeling(self, T: nx.DiGraph) -> nx.DiGraph:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(T))
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {
            node: self._max_dict_value(shortest_paths[node]) + 1 for node in T.nodes()
        }
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _max_depth_dataset_labeling(self, dataset: List[Cascade]) -> List[Cascade]:
        return [(self._max_depth_labeling(tree), target) for tree, target in dataset]

    # ---------- k-means-on-a-statistic labeling ----------

    def _kmeans_label_by_statistic(
        self, T: nx.DiGraph, stat_fn: Callable[[nx.DiGraph, object], float], k: int
    ) -> nx.DiGraph:
        nodes = list(T.nodes())
        stats = [stat_fn(T, node) for node in nodes]
        clusters = kMeans1d(stats, k, 10).assignments
        new_attribs = {node: clusters[i] + 1 for i, node in enumerate(nodes)}
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _tot_leaves_kmeans_node_labeling(self, T: nx.DiGraph, k: int) -> nx.DiGraph:
        def leaf_descendant_count(tree, node):
            return len([d for d in nx.descendants(tree, node) if tree.out_degree(d) == 0])

        return self._kmeans_label_by_statistic(T, leaf_descendant_count, k)

    def _tot_leaves_kmeans(self, dataset: List[Cascade]) -> List[Cascade]:
        return [
            (self._tot_leaves_kmeans_node_labeling(tree, 4), target) for tree, target in dataset
        ]

    def _descendants_kmeans(self, T: nx.DiGraph, k: int) -> nx.DiGraph:
        return self._kmeans_label_by_statistic(
            T, lambda tree, node: len(nx.descendants(tree, node)), k
        )

    def _cascade_based_descendant_kmeans_ds(self, dataset: List[Cascade]) -> List[Cascade]:
        return [(self._descendants_kmeans(tree, 4), target) for tree, target in dataset]

    def _set_label_kmeans_out_deg(self, T: nx.DiGraph, k: int) -> nx.DiGraph:
        return self._kmeans_label_by_statistic(T, lambda tree, node: tree.out_degree(node), k)

    def _cascade_based_kmeans(self, dataset: List[Cascade]) -> List[Cascade]:
        return [(self._set_label_kmeans_out_deg(tree, 4), target) for tree, target in dataset]

    # ---------- graph-based (per-user) labeling ----------

    def _graph_based_kmeans(self, dataset: List[Cascade]) -> List[Cascade]:
        freq_per_user = self._participation_count_per_user(dataset)
        user_cluster = self._bucket_from_values(freq_per_user, 4)
        new_dataset = []
        for tree, target in dataset:
            labeled = nx.DiGraph()
            labeled.add_nodes_from(tree.nodes())
            labeled.add_edges_from(tree.edges())
            new_attribs = {
                node: user_cluster[tree.nodes[node]["label"]] + 1 for node in tree.nodes()
            }
            nx.set_node_attributes(labeled, new_attribs, name="label")
            new_dataset.append((labeled, target))
        return new_dataset

    @staticmethod
    def _participation_count_per_user(dataset: List[Cascade]) -> Dict:
        user_appearance: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                user_appearance[user] = user_appearance.get(user, 0) + 1
        return user_appearance

    def _bucket_from_values(self, user_val_mapping: Dict, k: int) -> Dict:
        users = list(user_val_mapping.keys())
        vals = [user_val_mapping[u] for u in users]
        clusters = kMeans1d(vals, k, 10).assignments
        return dict(zip(users, clusters))

    # ---------- good/bad-user frequency labeling ----------

    def _good_bad_users_freq_participation(self, dataset: List[Cascade]) -> List[Cascade]:
        tot_cnt_per_user = self._frequency_participation_per_veracity(dataset)
        return [
            (self._good_bad_user_tree_node_labeling(tree, tot_cnt_per_user), self.target_to_int(target))
            for tree, target in dataset
        ]

    @staticmethod
    def _good_bad_user_tree_node_labeling(T: nx.DiGraph, dic: Dict) -> nx.DiGraph:
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {}
        for node in T.nodes():
            score = dic[T.nodes[node]["label"]]
            if score <= -1:
                new_attribs[node] = 1
            elif score >= 1:
                new_attribs[node] = 2
            else:
                new_attribs[node] = 3
        nx.set_node_attributes(res, new_attribs, name="label")
        res.add_edges_from(T.edges())
        return res

    @staticmethod
    def _frequency_participation_per_veracity(dataset: List[Cascade]) -> Dict:
        freq_per_user: Dict = {}
        for tree, target in dataset:
            delta = 1 if target == "true" else -1
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                freq_per_user[user] = freq_per_user.get(user, 0) + delta
        return freq_per_user

    # ---------- mean / log-binned out-degree labeling ----------

    def _mean_out_deg_dataset_labeling(self, dataset: List[Cascade]) -> List[Cascade]:
        mean_out_deg_per_user = self._mean_out_deg_per_user(dataset)
        return [
            (self._add_nodes_according_to_dict(tree, mean_out_deg_per_user), target)
            for tree, target in dataset
        ]

    def _log_bin_mean_out_deg(self, dataset: List[Cascade]) -> List[Cascade]:
        return self._mean_out_deg_dataset_labeling(dataset)

    def _log_bin_out_deg_sum(self, dataset: List[Cascade]) -> List[Cascade]:
        out_deg_sum_per_user = self._tot_out_deg_sum(dataset)
        return [
            (self._add_nodes_according_to_dict(tree, out_deg_sum_per_user), target)
            for tree, target in dataset
        ]

    def _add_nodes_according_to_dict(self, T: nx.DiGraph, dic: Dict) -> nx.DiGraph:
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {
            node: self._log_bin_bucket(int(dic[T.nodes[node]["label"]])) for node in T.nodes()
        }
        nx.set_node_attributes(res, new_attribs, name="label")
        res.add_edges_from(T.edges())
        return res

    @staticmethod
    def _tot_out_deg_sum(dataset: List[Cascade]) -> Dict:
        out_deg_sum: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                out_deg_sum[user] = out_deg_sum.get(user, 0) + tree.out_degree(node)
        return out_deg_sum

    @staticmethod
    def _mean_out_deg_per_user(dataset: List[Cascade]) -> Dict:
        tot_out_deg: Dict = {}
        tot_count: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                tot_out_deg[user] = tot_out_deg.get(user, 0) + tree.out_degree(node)
                tot_count[user] = tot_count.get(user, 0) + 1
        return {user: tot_out_deg[user] / tot_count[user] for user in tot_out_deg}

    @staticmethod
    def _tot_out_deg_per_user(dataset: List[Cascade]) -> Dict:
        """FIX: original had the if/else branches inverted (`+=` on a
        missing key raised KeyError). Corrected here."""
        tot_out_deg: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                tot_out_deg[user] = tot_out_deg.get(user, 0) + tree.out_degree(node)
        return tot_out_deg

    @staticmethod
    def _find_median(arr: List[float]) -> float:
        n = len(arr)
        if n % 2 == 1:
            return arr[n // 2]
        return (arr[n // 2 - 1] + arr[n // 2]) / 2

    def _median_out_deg_per_user(self, dataset: List[Cascade]) -> Dict:
        user_out_degs: Dict = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"]
                user_out_degs.setdefault(user, []).append(tree.out_degree(node))
        return {user: self._find_median(sorted(degs)) for user, degs in user_out_degs.items()}

    @staticmethod
    def _log_bin_bucket(n: int) -> int:
        if n == 0:
            return 1
        return int(np.floor(np.log10(n))) + 1

    def _out_deg_log_bin_labeling_tree(self, T: nx.DiGraph) -> nx.DiGraph:
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {
            node: (1 if T.out_degree(node) == 0 else self._log_bin_bucket(T.out_degree(node)))
            for node in T.nodes()
        }
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _node_labeling_cascades_log_bin(self, dataset: List[Cascade]) -> List[Cascade]:
        return self._node_labeling_cascades(dataset, self._out_deg_log_bin_labeling_tree)

    @staticmethod
    def _node_labeling_cascades(dataset: List[Cascade], node_labeling_method: Callable) -> List[Cascade]:
        return [(node_labeling_method(tree), target) for tree, target in dataset]

    # ---------- out-degree-statistics labeling ----------

    @staticmethod
    def _compute_max_out_deg(T: nx.DiGraph) -> int:
        return max(T.out_degree(node) for node in T.nodes())

    def _find_true_false_mean_val(self, dataset: List[Cascade]) -> Tuple[float, float]:
        true_max_out_deg = [
            self._compute_max_out_deg(tree) for tree, target in dataset if target == "true"
        ]
        false_max_out_deg = [
            self._compute_max_out_deg(tree) for tree, target in dataset if target != "true"
        ]
        mean_true = sum(true_max_out_deg) / len(true_max_out_deg) if true_max_out_deg else 0
        mean_false = sum(false_max_out_deg) / len(false_max_out_deg) if false_max_out_deg else 0
        return mean_true, mean_false

    def _out_deg_stats_tree_labeling(self, dataset: List[Cascade]) -> List[Cascade]:
        true_mean_out_deg, false_mean_out_deg = self._find_true_false_mean_val(dataset)
        new_dataset = []
        for tree, target in dataset:
            g = nx.DiGraph()
            g.add_nodes_from(tree.nodes())
            new_attribs = {}
            for node in tree.nodes():
                if tree.out_degree(node) < true_mean_out_deg:
                    new_attribs[node] = 1
                elif tree.out_degree(node) > false_mean_out_deg:
                    new_attribs[node] = 2
                else:
                    new_attribs[node] = 3
            g.add_edges_from(tree.edges())
            nx.set_node_attributes(g, new_attribs, name="label")
            new_dataset.append((g, target))
        return new_dataset
