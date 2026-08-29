from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    def __init__(self, trainingSetDataSet, testSetDataSet, random_state=42):
        X_train, y_train = self._split_features_labels(trainingSetDataSet)
        X_test, y_test = self._split_features_labels(testSetDataSet)

        self.results = [
            self._train_and_evaluate(
                LogisticRegression(max_iter=5000, solver='saga', random_state=random_state),
                X_train,
                y_train,
                X_test,
                y_test,
            ),
            self._train_and_evaluate(
                GradientBoostingClassifier(random_state=random_state),
                X_train,
                y_train,
                X_test,
                y_test,
            ),
        ]

    @staticmethod
    def _split_features_labels(dataset):
        X = [vec for vec, _ in dataset]
        y = [label for _, label in dataset]
        return X, y

    @staticmethod
    def _train_and_evaluate(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        return accuracy, report