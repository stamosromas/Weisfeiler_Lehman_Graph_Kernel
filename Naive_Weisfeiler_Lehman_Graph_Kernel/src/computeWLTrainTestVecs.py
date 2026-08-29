from wlGraphKernel import WLGraphKernel
from trainingAndTestSetSplitter import trainingSetTestSetSplitter


class computeWLTrainingTestSetVecs:
    def __init__(self, dataset, h, perc, maxTotalNodes):
        splitObj = trainingSetTestSetSplitter(dataset, perc, maxTotalNodes)
        self.trainingset = splitObj.trainingSet
        self.testset = splitObj.testSet

        if len(self.trainingset) == 0:
            raise ValueError("Training set cannot be empty.")

        self.h = h
        self.kernel = WLGraphKernel(h)

        self.trainingSetDataSet = []
        self.testSetDataSet = []

        self.main()

    def main(self):
        trainGraphs = [elem[0].to_undirected() for elem in self.trainingset]
        trainTargets = [elem[1] for elem in self.trainingset]

        trainVecs = self.kernel.fit(trainGraphs)
        self.trainingSetDataSet = list(zip(trainVecs.tolist(), trainTargets))

        if len(self.testset) > 0:
            testGraphs = [elem[0].to_undirected() for elem in self.testset]
            testTargets = [elem[1] for elem in self.testset]
            testVecs = self.kernel.transform(testGraphs)
            self.testSetDataSet = list(zip(testVecs.tolist(), testTargets))
        else:
            self.testSetDataSet = []