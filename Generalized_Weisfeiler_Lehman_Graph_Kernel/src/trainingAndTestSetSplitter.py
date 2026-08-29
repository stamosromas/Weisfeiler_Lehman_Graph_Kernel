"""
Splits a (tree, label) dataset into class-balanced train/test sets.
"""

import random
from typing import List, Tuple

import networkx as nx

Cascade = Tuple[nx.DiGraph, int]


class trainingSetTestSetSplitter:
    def __init__(self, dataset: List[Cascade], perc: float, maxSize: int, seed: int = None):
        self.rng = random.Random(seed)
        self.dataset = self._filter_by_size(dataset, maxSize)

        half_n = len(self.dataset) // 2
        self.target_per_class = int(perc * half_n)

        self.trainingSet: List[Cascade] = []
        self.testSet: List[Cascade] = []
        self.main()

    @staticmethod
    def _filter_by_size(dataset: List[Cascade], max_size: int) -> List[Cascade]:
        return [elem for elem in dataset if len(elem[0].nodes()) <= max_size]

    def _split_by_class(self, target_label: int) -> Tuple[List[Cascade], List[Cascade]]:
        elems = [elem for elem in self.dataset if elem[1] == target_label]
        self.rng.shuffle(elems)
        if len(elems) < self.target_per_class:
            print(
                f"Warning: requested {self.target_per_class} training examples for "
                f"class {target_label}, but only {len(elems)} are available after "
                f"filtering; using all of them for training."
            )
        return elems[: self.target_per_class], elems[self.target_per_class :]

    def main(self) -> None:
        true_train, true_test = self._split_by_class(1)
        false_train, false_test = self._split_by_class(0)

        self.trainingSet = true_train + false_train
        self.testSet = true_test + false_test

        self.rng.shuffle(self.trainingSet)
        self.rng.shuffle(self.testSet)
