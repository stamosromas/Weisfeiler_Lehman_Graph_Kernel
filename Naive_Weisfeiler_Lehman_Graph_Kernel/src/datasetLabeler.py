import networkx as nx
import numpy as np

from kMeansIn1d import kMeans1d


class nodeLabelingOfDataSet:

    _METHODS = {
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

    def __init__(self, dataSet, method):
        handler_name = self._METHODS.get(method, "_mean_out_deg_per_user_dataset")
        self.dataSet = getattr(self, handler_name)(dataSet)

    @staticmethod
    def target_to_int(target):
        return 1 if target == "true" or target == 1 else 0

    # ---------- simple label assignment ----------

    def _set_label_one_to_all_nodes(self, dataset):
        new_dataset = []
        for tree, target in dataset:
            g = nx.DiGraph()
            for node in tree.nodes():
                g.add_node(node, label=1)
            g.add_edges_from(tree.edges())
            new_dataset.append((g, self.target_to_int(target)))
        return new_dataset

    def _mean_out_deg_per_user_dataset(self, dataset):
        return self._mean_out_deg_dataset_labeling(dataset)

    # ---------- clustering-based user classification ----------

    @staticmethod
    def _classify_to_cluster(vals):
        if all(v == 0 for v in vals):
            return 1 if len(vals) > 2 else 2
        mean_val = float(sum(vals)) / len(vals)
        if mean_val < 0.3:
            return 2
        if mean_val < 1:
            return 3
        return 4

    @staticmethod
    def _label_tree_from_dict(T, label_map):
        res = nx.DiGraph()
        for node in T.nodes():
            node_label = T.nodes[node]["label"] if hasattr(T, "nodes") else T.node[node]["label"]
            res.add_node(node, label=label_map[node_label])
        res.add_edges_from(T.edges())
        return res

    def _create_heuristic_dataset(self, dataset):
        users_out_deg = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                if user not in users_out_deg:
                    users_out_deg[user] = []
                users_out_deg[user].append(tree.out_degree(node))

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
    def _max_dict_value(d):
        vals = d.values()
        return max(vals) if vals else 0

    def _root_deepest_leaf_node_labeling(self, T):
        in_degs = T.in_degree()
        in_degs_dict = dict(in_degs) if not isinstance(in_degs, dict) else in_degs
        root = next(n for n, d in in_degs_dict.items() if d == 0)
        
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

    def _root_deepest_leaf_node_labeling_dataset(self, dataset):
        return [
            (self._root_deepest_leaf_node_labeling(tree), self.target_to_int(target))
            for tree, target in dataset
        ]

    def _max_depth_labeling(self, T):
        shortest_paths = dict(nx.all_pairs_shortest_path_length(T))
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {
            node: self._max_dict_value(shortest_paths[node]) + 1 for node in T.nodes()
        }
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _max_depth_dataset_labeling(self, dataset):
        return [
            (self._max_depth_labeling(tree), self.target_to_int(target))
            for tree, target in dataset
        ]

    # ---------- k-means-on-a-statistic labeling ----------

    def _kmeans_label_by_statistic(self, T, stat_fn, k):
        nodes = list(T.nodes())
        stats = [stat_fn(T, node) for node in nodes]
        clusters = kMeans1d(stats, k, 10).assignments
        new_attribs = {node: clusters[i] + 1 for i, node in enumerate(nodes)}
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _tot_leaves_kmeans_node_labeling(self, T, k):
        def leaf_descendant_count(tree, node):
            return len([d for d in nx.descendants(tree, node) if tree.out_degree(d) == 0])

        return self._kmeans_label_by_statistic(T, leaf_descendant_count, k)

    def _tot_leaves_kmeans(self, dataset):
        return [
            (self._tot_leaves_kmeans_node_labeling(tree, 4), self.target_to_int(target))
            for tree, target in dataset
        ]

    def _descendants_kmeans(self, T, k):
        return self._kmeans_label_by_statistic(
            T, lambda tree, node: len(nx.descendants(tree, node)), k
        )

    def _cascade_based_descendant_kmeans_ds(self, dataset):
        return [
            (self._descendants_kmeans(tree, 4), self.target_to_int(target))
            for tree, target in dataset
        ]

    def _set_label_kmeans_out_deg(self, T, k):
        return self._kmeans_label_by_statistic(T, lambda tree, node: tree.out_degree(node), k)

    def _cascade_based_kmeans(self, dataset):
        return [
            (self._set_label_kmeans_out_deg(tree, 4), self.target_to_int(target))
            for tree, target in dataset
        ]

    # ---------- graph-based (per-user) labeling ----------

    def _graph_based_kmeans(self, dataset):
        freq_per_user = self._participation_count_per_user(dataset)
        user_cluster = self._bucket_from_values(freq_per_user, 4)
        new_dataset = []
        for tree, target in dataset:
            labeled = nx.DiGraph()
            labeled.add_nodes_from(tree.nodes())
            labeled.add_edges_from(tree.edges())
            new_attribs = {}
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                new_attribs[node] = user_cluster[user] + 1
            nx.set_node_attributes(labeled, new_attribs, name="label")
            new_dataset.append((labeled, self.target_to_int(target)))
        return new_dataset

    @staticmethod
    def _participation_count_per_user(dataset):
        user_appearance = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                user_appearance[user] = user_appearance.get(user, 0) + 1
        return user_appearance

    def _bucket_from_values(self, user_val_mapping, k):
        users = list(user_val_mapping.keys())
        vals = [user_val_mapping[u] for u in users]
        clusters = kMeans1d(vals, k, 10).assignments
        return dict(zip(users, clusters))

    # ---------- good/bad-user frequency labeling ----------

    def _good_bad_users_freq_participation(self, dataset):
        tot_cnt_per_user = self._frequency_participation_per_veracity(dataset)
        return [
            (self._good_bad_user_tree_node_labeling(tree, tot_cnt_per_user), self.target_to_int(target))
            for tree, target in dataset
        ]

    @staticmethod
    def _good_bad_user_tree_node_labeling(T, dic):
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {}
        for node in T.nodes():
            user = T.nodes[node]["label"] if hasattr(T, "nodes") else T.node[node]["label"]
            score = dic[user]
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
    def _frequency_participation_per_veracity(dataset):
        freq_per_user = {}
        for tree, target in dataset:
            delta = 1 if target == "true" or target == 1 else -1
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                freq_per_user[user] = freq_per_user.get(user, 0) + delta
        return freq_per_user

    # ---------- mean / log-binned out-degree labeling ----------

    def _mean_out_deg_dataset_labeling(self, dataset):
        mean_out_deg_per_user = self._mean_out_deg_per_user(dataset)
        return [
            (self._add_nodes_according_to_dict(tree, mean_out_deg_per_user), self.target_to_int(target))
            for tree, target in dataset
        ]

    def _log_bin_mean_out_deg(self, dataset):
        return self._mean_out_deg_dataset_labeling(dataset)

    def _log_bin_out_deg_sum(self, dataset):
        out_deg_sum_per_user = self._tot_out_deg_sum(dataset)
        return [
            (self._add_nodes_according_to_dict(tree, out_deg_sum_per_user), self.target_to_int(target))
            for tree, target in dataset
        ]

    def _add_nodes_according_to_dict(self, T, dic):
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {}
        for node in T.nodes():
            user = T.nodes[node]["label"] if hasattr(T, "nodes") else T.node[node]["label"]
            new_attribs[node] = self._log_bin_bucket(int(dic[user]))
        nx.set_node_attributes(res, new_attribs, name="label")
        res.add_edges_from(T.edges())
        return res

    @staticmethod
    def _tot_out_deg_sum(dataset):
        out_deg_sum = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                out_deg_sum[user] = out_deg_sum.get(user, 0) + tree.out_degree(node)
        return out_deg_sum

    @staticmethod
    def _mean_out_deg_per_user(dataset):
        tot_out_deg = {}
        tot_count = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                tot_out_deg[user] = tot_out_deg.get(user, 0) + tree.out_degree(node)
                tot_count[user] = tot_count.get(user, 0) + 1
        return {user: float(tot_out_deg[user]) / tot_count[user] for user in tot_out_deg}

    @staticmethod
    def _tot_out_deg_per_user(dataset):
        tot_out_deg = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                tot_out_deg[user] = tot_out_deg.get(user, 0) + tree.out_degree(node)
        return tot_out_deg

    @staticmethod
    def _find_median(arr):
        n = len(arr)
        if n % 2 == 1:
            return arr[n // 2]
        return float(arr[n // 2 - 1] + arr[n // 2]) / 2.0

    def _median_out_deg_per_user(self, dataset):
        user_out_degs = {}
        for tree, _ in dataset:
            for node in tree.nodes():
                user = tree.nodes[node]["label"] if hasattr(tree, "nodes") else tree.node[node]["label"]
                if user not in user_out_degs:
                    user_out_degs[user] = []
                user_out_degs[user].append(tree.out_degree(node))
        return {user: self._find_median(sorted(degs)) for user, degs in user_out_degs.items()}

    @staticmethod
    def _log_bin_bucket(n):
        if n == 0:
            return 1
        return int(np.floor(np.log10(n))) + 1

    def _out_deg_log_bin_labeling_tree(self, T):
        res = nx.DiGraph()
        res.add_nodes_from(T.nodes())
        new_attribs = {
            node: (1 if T.out_degree(node) == 0 else self._log_bin_bucket(T.out_degree(node)))
            for node in T.nodes()
        }
        res.add_edges_from(T.edges())
        nx.set_node_attributes(res, new_attribs, name="label")
        return res

    def _node_labeling_cascades_log_bin(self, dataset):
        return self._node_labeling_cascades(dataset, self._out_deg_log_bin_labeling_tree)

    def _node_labeling_cascades(self, dataset, node_labeling_method):
        return [(node_labeling_method(tree), self.target_to_int(target)) for tree, target in dataset]

    # ---------- out-degree-statistics labeling ----------

    @staticmethod
    def _compute_max_out_deg(T):
        return max(T.out_degree(node) for node in T.nodes())

    def _find_true_false_mean_val(self, dataset):
        true_max_out_deg = [
            self._compute_max_out_deg(tree) for tree, target in dataset if target == "true" or target == 1
        ]
        false_max_out_deg = [
            self._compute_max_out_deg(tree) for tree, target in dataset if target != "true" and target != 1
        ]
        mean_true = float(sum(true_max_out_deg)) / len(true_max_out_deg) if true_max_out_deg else 0.0
        mean_false = float(sum(false_max_out_deg)) / len(false_max_out_deg) if false_max_out_deg else 0.0
        return mean_true, mean_false

    def _out_deg_stats_tree_labeling(self, dataset):
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
            new_dataset.append((g, self.target_to_int(target)))
        return new_dataset