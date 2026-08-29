import os
import sys

from computeWLTrainTestVecs import computeWLTrainingTestSetVecs
from datasetLabeler import nodeLabelingOfDataSet
from datasetReader import readDataSetFromFullPath
from modelTrainer import ModelTrainer

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Usage: python mainWL.py <h> <trainingPercentage> <maxTotalNodes> <labeling_method>")

    h = int(sys.argv[1])
    percentageOfTrainingAndTestSet = float(sys.argv[2])
    maxTotalNodesInDataSet = int(sys.argv[3])
    labeling_method = sys.argv[4]

    dataPath = os.path.join("..", "..", "data", "label.txt")
    treePath = os.path.join("..", "..", "data", "tree")
    resultsPath = os.path.join("..", "Results", "WL_" + "_".join(sys.argv[1:]))

    os.makedirs(resultsPath, exist_ok=True)

    readDataSetObj = readDataSetFromFullPath(dataPath, treePath)
    dataSet = readDataSetObj.dataSet

    nodeLabelingObj = nodeLabelingOfDataSet(dataSet, method=labeling_method)
    newDataSet = nodeLabelingObj.dataSet

    experimentObject = computeWLTrainingTestSetVecs(
        newDataSet, h, percentageOfTrainingAndTestSet, maxTotalNodesInDataSet
    )

    trainingSetPath = os.path.join(resultsPath, "trainingSet.txt")
    testSetPath = os.path.join(resultsPath, "testSet.txt")

    with open(trainingSetPath, "w") as output:
        output.write(str(experimentObject.trainingSetDataSet))

    with open(testSetPath, "w") as output:
        output.write(str(experimentObject.testSetDataSet))

    trainer = ModelTrainer(experimentObject.trainingSetDataSet, experimentObject.testSetDataSet)
    results = trainer.results

    resultsFilePath = os.path.join(resultsPath, "results.txt")

    with open(resultsFilePath, "w") as f:
        f.write(f"Accuracy for Logistic Regression: {results[0][0]}\n\n")
        f.write(f"Classification Report for Logistic Regression:\n{results[0][1]}")
        f.write("\n\n\n--------------------------------------------------------\n\n\n")
        f.write(f"Accuracy for Gradient Boosting Trees: {results[1][0]}\n\n")
        f.write(f"Classification Report for Gradient Boosting Trees:\n{results[1][1]}")

    print("\n" + "=" * 60)
    print(f"EXPERIMENT RESULTS: WL_{'_'.join(sys.argv[1:])}")
    print("=" * 60)

    # models = ["Logistic Regression", "Gradient Boosting Trees"]
    # for name, (acc, report) in zip(models, results):
    #     print(f"\nModel: {name}")
    #     print(f"Accuracy: {acc:.4f}")
    #     print("Classification Report:")
    #     print(report)
    #     print("-" * 60)

    print(f"\nResults successfully saved to: {resultsFilePath}\n")