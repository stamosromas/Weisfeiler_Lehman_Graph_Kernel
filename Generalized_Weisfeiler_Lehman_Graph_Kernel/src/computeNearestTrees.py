"""
Finds the unfolding tree(s) in `unfTrees` nearest to a query tree `T` under
tree edit distance, using node-count bucketing to avoid comparing against
every candidate.
"""

import sys
from typing import Dict, List

import networkx as nx
import numpy as np

from treeEditDistanceWithCosts import SDTedClass


class FindNearestTrees:
    def __init__(self, unfTrees: List[nx.DiGraph], T: nx.DiGraph, CostMatrix: np.ndarray):
        self.UnfoldingTrees = unfTrees
        self.T = T
        self.CostMatrix = CostMatrix
        self.UnfTreesBucketing = self._create_bucketing()
        self.minDistOfSameBuck = self._smallest_dist_in_same_bucket()
        self.res = self._main()

    def _create_bucketing(self) -> Dict[int, List[nx.DiGraph]]:
        buckets: Dict[int, List[nx.DiGraph]] = {}
        for unf_tree in self.UnfoldingTrees:
            buckets.setdefault(len(unf_tree.nodes()), []).append(unf_tree)
        return dict(sorted(buckets.items()))

    def _smallest_dist_in_same_bucket(self) -> float:
        bucket = self.UnfTreesBucketing.get(len(self.T.nodes()))
        if not bucket:
            return sys.maxsize
        return min(SDTedClass(tree, self.T, self.CostMatrix).res for tree in bucket)

    @staticmethod
    def _range_of_list(ls, center: float, diff: float) -> List[int]:
        window = set(range(int(center - diff), int(center + diff) + 1))
        return sorted(set(ls) & window)

    @staticmethod
    def _nearest_bucket_sizes(sizes, target: int) -> List[int]:
        min_dist = min(abs(size - target) for size in sizes)
        return [size for size in sizes if abs(size - target) == min_dist]

    def _candidate_trees(self) -> List[nx.DiGraph]:
        d = self.minDistOfSameBuck
        if d == sys.maxsize:
            closest_buckets = self._nearest_bucket_sizes(
                self.UnfTreesBucketing.keys(), len(self.T.nodes())
            )
            candidates = []
            for bucket in closest_buckets:
                candidates += self.UnfTreesBucketing[bucket]
            return candidates

        interval = self._range_of_list(
            list(self.UnfTreesBucketing.keys()), len(self.T.nodes()), d
        )
        candidates = []
        for bucket_size in interval:
            candidates += self.UnfTreesBucketing[bucket_size]
        return candidates

    def _main(self) -> List[nx.DiGraph]:
        candidates = self._candidate_trees()
        min_dist = self.minDistOfSameBuck
        nearest: List[nx.DiGraph] = []
        for tree in candidates:
            dist = SDTedClass(self.T, tree, self.CostMatrix).res
            if dist < min_dist:
                min_dist = dist
                nearest = [tree]
            elif dist == min_dist:
                nearest.append(tree)
        return nearest
