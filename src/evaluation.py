# Ai assisted code:
# The evaluation section below was generated with the assistance of an ai
# and reviewed to use in this project.



from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.baseline import (
    always_negative_baseline,
    persistence_baseline,
    seasonal_baseline,
)
from src.data_loader import (
    add_target,
    load_all_stations,
    remove_missing_targets,
)
from src.features import (
    add_lag_features,
    remove_missing_features,
)
from src.models import (
    evaluate_thresholds,
    predict_with_threshold,
    train_logistic_regression,
    train_random_forest,
)
from src.split import (
    chronological_split,
    split_features_target,
)


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    scores,
) -> dict:

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "average_precision": average_precision_score(
            y_true,
            scores,
        ),
    }


def select_threshold(
    y_validation: pd.Series,
    probabilities,
) -> float:

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    results = evaluate_thresholds(
        y_validation,
        probabilities,
        thresholds,
    )

    selected_threshold = (
        results
        .sort_values("f1", ascending=False)
        .iloc[0]["threshold"]
    )

    return float(selected_threshold)


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_all_stations(DATA_DIR)
    df = add_target(df)
    df = remove_missing_targets(df)
    df = add_lag_features(df)
    df = remove_missing_features(df)

    train, validation, test = chronological_split(df)

    X_train, y_train = split_features_target(train)
    X_validation, y_validation = split_features_target(
        validation
    )
    X_test, y_test = split_features_target(test)

    results = {}
    confusion_matrices = {}

    # -------------------------
    # Baseline: always negative
    # -------------------------

    predictions = always_negative_baseline(y_test)

    results["Always negative"] = calculate_metrics(
        y_test,
        predictions,
        predictions,
    )

    confusion_matrices["Always negative"] = confusion_matrix(
        y_test,
        predictions,
    )

    # -------------------------
    # Baseline: persistence
    # -------------------------

    predictions = persistence_baseline(
        test["TXK"]
    )

    results["Persistence"] = calculate_metrics(
        y_test,
        predictions,
        predictions,
    )

    confusion_matrices["Persistence"] = confusion_matrix(
        y_test,
        predictions,
    )

    # -------------------------
    # Baseline: seasonal
    # -------------------------

    predictions = seasonal_baseline(
        test["MESS_DATUM"]
    )

    results["Seasonal"] = calculate_metrics(
        y_test,
        predictions,
        predictions,
    )

    confusion_matrices["Seasonal"] = confusion_matrix(
        y_test,
        predictions,
    )

    # -------------------------
    # Logistic Regression
    # -------------------------

    logistic_model = train_logistic_regression(
        X_train,
        y_train,
    )

    logistic_validation_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    logistic_threshold = select_threshold(
        y_validation,
        logistic_validation_probabilities,
    )

    logistic_test_probabilities = (
        logistic_model.predict_proba(
            X_test
        )[:, 1]
    )

    logistic_predictions = predict_with_threshold(
        logistic_test_probabilities,
        logistic_threshold,
    )

    results["Logistic Regression"] = calculate_metrics(
        y_test,
        logistic_predictions,
        logistic_test_probabilities,
    )

    confusion_matrices["Logistic Regression"] = (
        confusion_matrix(
            y_test,
            logistic_predictions,
        )
    )

    # -------------------------
    # Random Forest
    # -------------------------

    random_forest_model = train_random_forest(
        X_train,
        y_train,
    )

    rf_validation_probabilities = (
        random_forest_model.predict_proba(
            X_validation
        )[:, 1]
    )

    rf_threshold = select_threshold(
        y_validation,
        rf_validation_probabilities,
    )

    rf_test_probabilities = (
        random_forest_model.predict_proba(
            X_test
        )[:, 1]
    )

    rf_predictions = predict_with_threshold(
        rf_test_probabilities,
        rf_threshold,
    )

    results["Random Forest"] = calculate_metrics(
        y_test,
        rf_predictions,
        rf_test_probabilities,
    )

    confusion_matrices["Random Forest"] = (
        confusion_matrix(
            y_test,
            rf_predictions,
        )
    )

    # -------------------------
    # Results table
    # -------------------------

    results_df = pd.DataFrame(results).T

    results_df = results_df.round(4)

    print("\nModel comparison:")
    print(results_df)

    print(
        "\nLogistic Regression threshold:",
        logistic_threshold,
    )

    print(
        "Random Forest threshold:",
        rf_threshold,
    )

    print("\nConfusion matrices:")

    for name, matrix in confusion_matrices.items():
        print(f"\n{name}")
        print(matrix)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv"
    )

    # -------------------------
    # Metrics plot
    # -------------------------

    metrics_to_plot = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "average_precision",
    ]

    results_df[metrics_to_plot].plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(
        title="Metric",
        loc="lower right",
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_metrics.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Confusion matrix plot
    # -------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for axis, (name, matrix) in zip(
        axes,
        confusion_matrices.items(),
    ):
        image = axis.imshow(matrix)

        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")

        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])

        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    matrix[row, column],
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.7,
    )


    plt.savefig(
        OUTPUT_DIR / "confusion_matrices.png",
        dpi=150,
    )

    plt.close()


if __name__ == "__main__":
    main()