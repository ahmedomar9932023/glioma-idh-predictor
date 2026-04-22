from __future__ import annotations

import json
import warnings

import bootstrap  # noqa: F401
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from config import MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, RESULTS_DIR
from pipeline_components import FeatureNameCleaner

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_processed() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_csv(PROCESSED_DIR / "source2_expression_matched.csv.gz")
    labels = pd.read_csv(PROCESSED_DIR / "source2_labels_metadata.csv")
    data = labels[["SAMPLE_ID", "PATIENT_ID", "IDH_status"]].merge(X, on="SAMPLE_ID", validate="one_to_one")
    y = data["IDH_status"].astype(int)
    features = data.drop(columns=["SAMPLE_ID", "PATIENT_ID", "IDH_status"])
    return features, y, data[["SAMPLE_ID", "PATIENT_ID", "IDH_status"]]


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_names", FeatureNameCleaner()),
            ("imputer", SimpleImputer(strategy="median")),
            ("log2", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("variance", VarianceThreshold(threshold=1e-8)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=500)),
            ("model", model),
        ]
    )


def candidate_models() -> dict:
    models = {
        "logistic_regression": (
            build_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
            {
                "select__k": [100, 500, 1000],
                "model__C": [0.01, 0.1, 1.0],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"],
            },
        ),
        "linear_svm": (
            build_pipeline(SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            {"select__k": [100, 500, 1000], "model__C": [0.01, 0.1, 1.0]},
        ),
        "rbf_svm": (
            build_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            {"select__k": [100, 500], "model__C": [0.1, 1.0], "model__gamma": ["scale", 0.001]},
        ),
        "random_forest": (
            build_pipeline(
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            {"select__k": [500, 1000], "model__max_depth": [3, 5, None], "model__min_samples_leaf": [2, 5]},
        ),
        "extra_trees": (
            build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            {"select__k": [500, 1000], "model__max_depth": [3, 5, None], "model__min_samples_leaf": [2, 5]},
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = (
            build_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                )
            ),
            {"select__k": [500, 1000], "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = (
            build_pipeline(LGBMClassifier(objective="binary", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            {"select__k": [500, 1000], "model__num_leaves": [7, 15], "model__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        pass
    return models


def metrics(y_true, y_pred, y_score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def score_estimator(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def selected_feature_names(pipe: Pipeline) -> np.ndarray:
    names = np.array(pipe.named_steps["feature_names"].get_feature_names_out())
    variance_mask = pipe.named_steps["variance"].get_support()
    names = names[variance_mask]
    select_mask = pipe.named_steps["select"].get_support()
    return names[select_mask]


def top_genes(pipe: Pipeline) -> pd.DataFrame:
    names = selected_feature_names(pipe)
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
        signed = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        signed = importance
    else:
        scores = pipe.named_steps["select"].scores_[pipe.named_steps["select"].get_support()]
        importance = np.nan_to_num(scores)
        signed = importance
    order = np.argsort(importance)[::-1]
    return pd.DataFrame({"gene": names[order], "importance": importance[order], "signed_value": signed[order]})


def plot_precision_recall(y_true, y_score, path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#0f766e", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Test Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, ids = load_processed()
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_results = {}
    best_estimators = {}

    for name, (pipe, grid) in candidate_models().items():
        search = GridSearchCV(
            pipe,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_estimators[name] = best
        train_score = score_estimator(best, X_train)
        train_pred = best.predict(X_train)
        cv_scores = cross_validate(
            best,
            X_train,
            y_train,
            scoring=["roc_auc", "average_precision", "balanced_accuracy", "f1"],
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        model_results[name] = {
            "best_params": search.best_params_,
            "grid_best_cv_roc_auc": float(search.best_score_),
            "train_metrics_on_refit": metrics(y_train, train_pred, train_score),
            "cv_mean": {k.replace("test_", ""): float(v.mean()) for k, v in cv_scores.items() if k.startswith("test_")},
            "cv_std": {k.replace("test_", ""): float(v.std()) for k, v in cv_scores.items() if k.startswith("test_")},
            "cv_train_mean": {k.replace("train_", ""): float(v.mean()) for k, v in cv_scores.items() if k.startswith("train_")},
        }

    best_cv_auc = max(result["cv_mean"]["roc_auc"] for result in model_results.values())
    eligible = [
        name
        for name, result in model_results.items()
        if result["cv_mean"]["roc_auc"] >= best_cv_auc - 0.005
    ]

    def generalization_key(name: str) -> tuple[float, float, float]:
        result = model_results[name]
        balanced_gap = abs(result["train_metrics_on_refit"]["balanced_accuracy"] - result["cv_mean"]["balanced_accuracy"])
        auc_gap = abs(result["train_metrics_on_refit"]["roc_auc"] - result["cv_mean"]["roc_auc"])
        return (
            result["cv_mean"]["balanced_accuracy"],
            -balanced_gap,
            -auc_gap,
        )

    best_name = sorted(eligible, key=generalization_key, reverse=True)[0]
    final_model = best_estimators[best_name]

    test_score = score_estimator(final_model, X_test)
    test_pred = final_model.predict(X_test)
    test_metrics = metrics(y_test, test_pred, test_score)
    cm = confusion_matrix(y_test, test_pred)

    final_report = {
        "selection_rule": "Models within 0.005 of the best inner-CV ROC AUC are ranked by CV balanced accuracy, then smaller train/CV gaps. This intentionally avoids choosing a perfect-training model when a similarly accurate model generalizes more honestly.",
        "chosen_model": best_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "model_comparison": model_results,
        "final_test_metrics": test_metrics,
        "final_test_confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "leakage_controls": [
            "Final test split is created before imputation, log transform, scaling, variance filtering, feature selection, and tuning.",
            "All preprocessing and SelectKBest feature selection are inside sklearn Pipeline and therefore fitted only on training folds during CV.",
            "Mutation data are used only to create IDH_status and never enter X.",
            "Patient-level leakage is avoided for Source 2 because there is one sample per patient in the matched dataset.",
            "Source 1 mutation matrix is excluded from predictors.",
        ],
    }

    joblib.dump(final_model, MODELS_DIR / "best_idh_expression_pipeline.joblib")
    ids_test.assign(y_true=y_test.values, y_pred=test_pred, y_score=test_score).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    pd.DataFrame(model_results).T.to_json(RESULTS_DIR / "model_comparison.json", indent=2)
    (REPORTS_DIR / "training_report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    genes = top_genes(final_model)
    genes.to_csv(RESULTS_DIR / "top_genes.csv", index=False)
    genes.head(30).sort_values("importance").plot.barh(x="gene", y="importance", figsize=(8, 8), legend=False, color="#2563eb")
    plt.xlabel("Model importance")
    plt.title(f"Top Genes: {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_genes.png", dpi=180)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, test_score, color="#1d4ed8")
    plt.title("Final Test ROC Curve")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=180)
    plt.close()
    plot_precision_recall(y_test, test_score, PLOTS_DIR / "precision_recall_curve.png")
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, cmap="Blues")
    plt.title("Final Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
