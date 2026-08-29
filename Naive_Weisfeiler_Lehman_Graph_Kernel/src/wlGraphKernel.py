import numpy as np

from wlGraph import WLGraph


class WLGraphListKernel:

    def __init__(self, graphList, h):
        self.graphList = self.toWLGraph(graphList)
        self.allLabels = [[] for i in range(len(graphList))]
        self.graphsToVec = []
        self.h = h

    def toWLGraph(self, listOfGraphs):
        return [WLGraph(graph) for graph in listOfGraphs]

    def computeMaxGraphLabel(self):
        maxLabel = 0
        for graph in self.graphList:
            maxLabel = max(maxLabel, graph.computeMaxLabel())
        return maxLabel

    def custom_sort(self, element):
        parts = element.split('.')
        return tuple(int(part) for part in parts)

    def labelCompression(self):
        multStrsMapping = {}
        enumId = self.computeMaxGraphLabel() + 1
        allMultStrs = []
        for graph in self.graphList:
            allMultStrs += graph.computeMultiLabelString()
        allMultStrs = sorted(list(set(allMultStrs)), key=self.custom_sort)
        for s in allMultStrs:
            multStrsMapping[s] = enumId
            enumId += 1
        return multStrsMapping

    def graphsRelabeling(self):
        multStrsEnum = self.labelCompression()
        for graph in self.graphList:
            graph.relabelingNodes(multStrsEnum)

    def addLabelsToMatrix(self):
        for i, graph in enumerate(self.graphList):
            self.allLabels[i] += graph.getGraphLabels()

    def computeFrequencies(self, arr, maxSize):
        res = np.zeros(maxSize)
        for elem in arr:
            res[elem - 1] += 1
        return res

    def findMaxInMatrix(self, mat):
        maxElem = 0
        for row in mat:
            maxElem = max(maxElem, max(row))
        return maxElem

    def mainFun(self):
        for i in range(self.h + 1):
            self.addLabelsToMatrix()
            self.graphsRelabeling()
        vecSize = self.findMaxInMatrix(self.allLabels)
        for i, row in enumerate(self.allLabels):
            self.graphsToVec.append(self.computeFrequencies(self.allLabels[i], vecSize))


class WLGraphKernel:

    UNSEEN_LABEL = WLGraph.UNSEEN_LABEL

    def __init__(self, h):
        if h < 0:
            raise ValueError("h must be non-negative.")

        self.h = h
        self.vocab = {}
        self.vecSize = 0
        self.fitted = False

    def _toWLGraphs(self, graphs):
        return [WLGraph(g) for g in graphs]

    def _customSort(self, element):
        parts = element.split('.')
        return tuple(int(p) for p in parts)

    def _computeMaxGraphLabel(self, wlGraphs):
        return max(g.computeMaxLabel() for g in wlGraphs)

    def _computeFrequencies(self, labels, size):
        res = np.zeros(size)
        for label in labels:
            if 1 <= label <= size:
                res[label - 1] += 1
        return res

    def fit(self, trainingGraphs):
        if not trainingGraphs:
            raise ValueError("Training graph list cannot be empty.")

        wlGraphs = self._toWLGraphs(trainingGraphs)
        allLabels = [[] for _ in wlGraphs]

        nextId = self._computeMaxGraphLabel(wlGraphs) + 1
        vocab = {}

        for _ in range(self.h + 1):
            for i, g in enumerate(wlGraphs):
                allLabels[i] += g.getGraphLabels()

            allMultStrs = set()
            for g in wlGraphs:
                allMultStrs.update(g.computeMultiLabelString())

            newStrs = sorted(allMultStrs - vocab.keys(), key=self._customSort)
            for s in newStrs:
                vocab[s] = nextId
                nextId += 1

            for g in wlGraphs:
                g.relabelingNodes(vocab)

        self.vocab = vocab
        self.vecSize = nextId - 1
        self.fitted = True

        return np.array([self._computeFrequencies(labels, self.vecSize) for labels in allLabels])

    def transform(self, graphs):
        if not self.fitted:
            raise RuntimeError("Call fit(trainingGraphs) before transform(...).")

        if not graphs:
            return np.zeros((0, self.vecSize))

        wlGraphs = self._toWLGraphs(graphs)
        allLabels = [[] for _ in wlGraphs]

        for _ in range(self.h + 1):
            for i, g in enumerate(wlGraphs):
                allLabels[i] += g.getGraphLabels()

            for g in wlGraphs:
                g.relabelingNodesTolerant(self.vocab)

        return np.array([self._computeFrequencies(labels, self.vecSize) for labels in allLabels])