"""
Reads a Twitter15/Twitter16-style rumor-detection dataset:
  * `labelFullPath`: a text file with lines like `true:tweetid` / `false:tweetid`
    (other labels, e.g. 'unverified'/'non-rumor' in the original Ma et al.
    dataset, are ignored — this pipeline is binary true/false only).
  * `treesFullPath`: a directory containing one file per tweet id, each line
    describing a propagation edge as `[parent_uid, parent_tweetid, t0]->[child_uid, child_tweetid, t1]`.
"""

import os
from typing import Dict, List, Tuple

import networkx as nx

Cascade = Tuple[nx.DiGraph, str]

KNOWN_BAD_TWEET_IDS = {
    "538975342011363328",
    "528822281972498432",
    "523123779124600833",
    "514517213022543872",
    "519868410599993344",
    "531607884220485632",
    "519929168864497664",
}


class readDataSetFromFullPath:
    def __init__(self, labelFullPath: str, treesFullPath: str):
        self.tweetIdVeracityStatus = self._read_veracity_per_tweet_id(labelFullPath)
        self.dataSet: List[Cascade] = self._create_dataset(treesFullPath)

    @staticmethod
    def _read_veracity_per_tweet_id(label_status_full_path: str) -> Dict[str, str]:
        tweet_id_veracity: Dict[str, str] = {}
        with open(label_status_full_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(":")
                if len(parts) < 2:
                    continue
                label, tweet_id = parts[0], parts[1]
                if label in ("true", "false"):
                    tweet_id_veracity[tweet_id] = label
        return tweet_id_veracity

    def _create_dataset(self, tree_folder_full_path: str) -> List[Cascade]:
        dataset: List[Cascade] = []
        for filename in os.listdir(tree_folder_full_path):
            tweet_id = filename.split(".")[0]
            if tweet_id in KNOWN_BAD_TWEET_IDS:
                continue
            if tweet_id not in self.tweetIdVeracityStatus:
                continue
            full_path = os.path.join(tree_folder_full_path, filename)
            try:
                tree = self._build_cascade_graph(full_path)
            except (IndexError, ValueError) as exc:
                print(f"Skipping malformed tree file {full_path}: {exc}")
                continue
            dataset.append((tree, self.tweetIdVeracityStatus[tweet_id]))
        return dataset

    @staticmethod
    def _strip_brackets(tokens: List[str]) -> List[str]:
        return [tok[1:-1] for tok in tokens]

    def _parse_edge_line(self, line: str) -> Tuple[List[str], List[str]]:
        line = line.rstrip("\n")
        parent_str, child_str = line.split("->")
        parent_fields = self._strip_brackets(parent_str.strip("][").split(", "))
        child_fields = self._strip_brackets(child_str.strip("][").split(", "))
        return parent_fields, child_fields

    def _parse_tree_file(self, full_path: str) -> List[Tuple[List[str], List[str]]]:
        with open(full_path, "r", encoding="utf-8") as f:
            return [self._parse_edge_line(line) for line in f if line.strip()]

    def _build_cascade_graph(self, file_path: str) -> nx.DiGraph:
        edges = self._parse_tree_file(file_path)

        res = nx.DiGraph()
        root_user = edges[0][1][0]
        res.add_node(1, label=root_user, time=0.0)
        user_to_node_id = {root_user: 1}
        node_id = 2

        for parent_fields, child_fields in edges[1:]:
            parent_user = parent_fields[0]
            child_user = child_fields[0]
            parent_node = user_to_node_id[parent_user]
            child_time = float(res.nodes[parent_node]["time"]) + float(child_fields[2])
            res.add_node(node_id, label=child_user, time=child_time)
            res.add_edge(parent_node, node_id)
            user_to_node_id[child_user] = node_id
            node_id += 1

        return res

    returnVeracityPerTweetId = _read_veracity_per_tweet_id
    createDataSet = _create_dataset
    fullGraphWithTimeFeatures = _build_cascade_graph
