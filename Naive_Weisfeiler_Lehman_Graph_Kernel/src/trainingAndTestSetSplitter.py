"""
Splits a (tree, label) dataset into class-balanced train/test sets.
"""

import random
import networkx as nx


class trainingSetTestSetSplitter:
    def __init__(self, dataset, perc, maxSize, seed=None):
        self.rng = random.Random(seed)
        self.dataset = self._filter_by_size(dataset, maxSize)

        half_n = len(self.dataset) // 2
        self.target_per_class = int(perc * half_n)

        self.trainingSet = []
        self.testSet = []
        self.main()

    @staticmethod
    def _filter_by_size(dataset, max_size):
        return [elem for elem in dataset if len(elem[0].nodes()) <= max_size]

    def _split_by_class(self, target_label):
        elems = [elem for elem in self.dataset if elem[1] == target_label]
        self.rng.shuffle(elems)
        if len(elems) < self.target_per_class:
            print(
                "Warning: requested %d training examples for class %d, "
                "but only %d are available after filtering; using all of them for training."
                % (self.target_per_class, target_label, len(elems))
            )
        return elems[: self.target_per_class], elems[self.target_per_class :]

    def main(self):
        true_train, true_test = self._split_by_class(1)
        false_train, false_test = self._split_by_class(0)

        self.trainingSet = true_train + false_train
        self.testSet = true_test + false_test

        self.rng.shuffle(self.trainingSet)
        self.rng.shuffle(self.testSet)