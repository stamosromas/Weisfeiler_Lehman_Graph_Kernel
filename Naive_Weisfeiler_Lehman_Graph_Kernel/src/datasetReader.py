import os
import networkx as nx

KNOWN_BAD_TWEET_IDS = set([
    "538975342011363328",
    "528822281972498432",
    "523123779124600833",
    "514517213022543872",
    "519868410599993344",
    "531607884220485632",
    "519929168864497664",
])


class readDataSetFromFullPath:
    def __init__(self, labelFullPath, treesFullPath):
        self.tweetIdVeracityStatus = self._read_veracity_per_tweet_id(labelFullPath)
        self.dataSet = self._create_dataset(treesFullPath)

    @staticmethod
    def _read_veracity_per_tweet_id(label_status_full_path):
        tweet_id_veracity = {}
        with open(label_status_full_path, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split(":")
                if len(parts) < 2:
                    continue
                label = parts[0]
                tweet_id = parts[1]
                if label in ("true", "false"):
                    tweet_id_veracity[tweet_id] = label
        return tweet_id_veracity

    def _create_dataset(self, tree_folder_full_path):
        dataset = []
        for filename in os.listdir(tree_folder_full_path):
            tweet_id = filename.split(".")[0]
            if tweet_id in KNOWN_BAD_TWEET_IDS:
                continue
            if tweet_id not in self.tweetIdVeracityStatus:
                continue
            full_path = os.path.join(tree_folder_full_path, filename)
            try:
                tree = self._build_cascade_graph(full_path)
            except (IndexError, ValueError, KeyError) as exc:
                print("Skipping malformed tree file %s: %s" % (full_path, exc))
                continue
            dataset.append((tree, self.tweetIdVeracityStatus[tweet_id]))
        return dataset

    @staticmethod
    def _strip_brackets(tokens):
        return [tok[1:-1] for tok in tokens]

    def _parse_edge_line(self, line):
        line = line.rstrip("\n")
        parent_str, child_str = line.split("->")
        parent_fields = self._strip_brackets(parent_str.strip("][").split(", "))
        child_fields = self._strip_brackets(child_str.strip("][").split(", "))
        return parent_fields, child_fields

    def _parse_tree_file(self, full_path):
        with open(full_path, "r") as f:
            return [self._parse_edge_line(line) for line in f if line.strip()]

    def _build_cascade_graph(self, file_path):
        edges = self._parse_tree_file(file_path)

        if not edges:
            raise ValueError("Tree file contains no edges")

        res = nx.DiGraph()

        root_parent_fields, root_child_fields = edges[0]
        root_user = root_child_fields[0]
        root_tweet_id = root_child_fields[1]

        res.add_node(1, label=root_user, time=0.0)

        tweet_to_node_id = {root_tweet_id: 1}
        node_id = 2

        for parent_fields, child_fields in edges[1:]:
            parent_user = parent_fields[0]
            parent_tweet_id = parent_fields[1]

            child_user = child_fields[0]
            child_tweet_id = child_fields[1]

            if parent_tweet_id in tweet_to_node_id:
                parent_node = tweet_to_node_id[parent_tweet_id]
            else:
                parent_node = 1

            child_time = float(res.nodes[parent_node]["time"]) + float(child_fields[2])

            res.add_node(node_id, label=child_user, time=child_time)
            res.add_edge(parent_node, node_id)

            tweet_to_node_id[child_tweet_id] = node_id
            node_id += 1

        return res

    def returnVeracityPerTweetId(self, label_status_full_path):
        return self._read_veracity_per_tweet_id(label_status_full_path)

    def createDataSet(self, tree_folder_full_path):
        return self._create_dataset(tree_folder_full_path)

    def fullGraphWithTimeFeatures(self, file_path):
        return self._build_cascade_graph(file_path)