import argparse
import os
from pathlib import Path

from computeTrainTestVecs import computeTrainingTestSetVecs
from datasetLabeler import nodeLabelingOfDataSet
from datasetReader import readDataSetFromFullPath
from modelTrainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one tree-edit-distance rumor-detection experiment.")
    parser.add_argument("clusters1UnfTree", type=int, help="k for 1-unfolding-tree clustering")
    parser.add_argument("clusters2UnfTree", type=int, help="k for 2-unfolding-tree clustering")
    parser.add_argument("percentageOfTrainingAndTestSet", type=float, help="fraction of each class used for training")
    parser.add_argument("maxTotalNodesInDataSet", type=int, help="drop cascades larger than this many nodes")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("THESIS_DATA_ROOT", "./data"),
        help="Directory containing label.txt and a tree/ subfolder (default: $THESIS_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=os.environ.get("THESIS_RESULTS_ROOT", "./Results"),
        help="Directory to write per-experiment result folders into (default: $THESIS_RESULTS_ROOT or ./Results)",
    )
    parser.add_argument(
        "--label-method",
        type=str,
        default="all-nodes-same-label",
        help="Node labeling strategy passed to nodeLabelingOfDataSet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    label_path = data_root / "label.txt"
    trees_path = data_root / "tree"

    read_dataset = readDataSetFromFullPath(str(label_path), str(trees_path))
    dataset = read_dataset.dataSet

    labeled = nodeLabelingOfDataSet(dataset, args.label_method).dataSet

    experiment = computeTrainingTestSetVecs(
        labeled,
        args.clusters1UnfTree,
        args.clusters2UnfTree,
        args.percentageOfTrainingAndTestSet,
        args.maxTotalNodesInDataSet,
    )

    run_id = "_".join(
        str(v)
        for v in (
            args.clusters1UnfTree,
            args.clusters2UnfTree,
            args.percentageOfTrainingAndTestSet,
            args.maxTotalNodesInDataSet,
        )
    )
    out_dir = Path(args.results_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "trainingSet.txt").write_text(str(experiment.trainingSetDataSet))
    (out_dir / "testSet.txt").write_text(str(experiment.testSetDataSet))

    trainer = ModelTrainer(experiment.trainingSetDataSet, experiment.testSetDataSet)
    (lr_accuracy, lr_report), (gbt_accuracy, gbt_report) = trainer.results

    results_text = (
        f"Accuracy for Logistic Regression: {lr_accuracy}\n\n"
        f"Classification Report for Logistic Regression:\n{lr_report}"
        "\n\n\n--------------------------------------------------------\n\n\n"
        f"Accuracy for Gradient Boosting Trees: {gbt_accuracy}\n\n"
        f"Classification Report for Gradient Boosting Trees:\n{gbt_report}"
    )
    (out_dir / "results.txt").write_text(results_text)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
