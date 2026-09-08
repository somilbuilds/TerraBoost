"""
train_model.py — Cover Type Explorer
Loads the UCI Covertype dataset, trains an XGBoost multi-class classifier,
evaluates performance, and saves the model for the web app.
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

# ── Cover-type class names (1-indexed in the dataset) ──────────────────────
CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [f"Wilderness_Area_{i}" for i in range(1, 5)] + [
    f"Soil_Type_{i}" for i in range(1, 41)
]


def load_dataset():
    """Load the Covertype dataset. Tries ucimlrepo first, falls back to sklearn."""
    print("Loading UCI Covertype dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features.values.astype(np.float32)
        y = covertype.data.targets.values.ravel().astype(np.int32)
        print(f"  Loaded via ucimlrepo: {X.shape[0]} samples, {X.shape[1]} features")
    except Exception as e:
        print(f"  ucimlrepo failed ({e}), falling back to sklearn...")
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int32)
        print(f"  Loaded via sklearn: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_and_evaluate():
    X, y = load_dataset()

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # XGBoost multi-class classifier
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )

    # XGBoost expects 0-indexed classes
    model.fit(X_train, y_train - 1)

    # Predict (convert back to 1-indexed for reporting)
    y_pred = model.predict(X_test) + 1

    # ── Metrics ──────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in range(1, 8)],
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'='*60}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"{'='*60}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{np.array(cm)}")

    # ── Save model & metrics ─────────────────────────────────────────────
    model.save_model("model.json")
    print("\nModel saved → model.json")

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved → metrics.json")


if __name__ == "__main__":
    train_and_evaluate()
