import argparse
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main(test_size, random_state):

    iris = load_iris()
    X = iris.data
    y = iris.target
    print(iris.feature_names, iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/iris_model.pkl")
    print("Model saved as outputs/iris_model.pkl")

    y_pred = model.predict(X_test)
    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("Confusion matrix image saved in outputs/confusion_matrix.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.test_size, args.random_state)
