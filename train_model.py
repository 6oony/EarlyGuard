import os
import glob
import json
import pickle
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANSOMWARE_DIR = "dataset/ransomware_calls"
BENIGN_DIR = "dataset/benign_calls"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NGRAM_MIN = 1
NGRAM_MAX = 3
MAX_FEATURES = 5000

TEST_SIZE = 0.2
RANDOM_STATE = 42

EARLY_STAGE_CALLS = 120
MIN_SYSCALLS = 20
CV_FOLDS = 5


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def read_syscalls_from_csv(filepath: str) -> List[str]:
    """
    Reads the first column from a CSV file and returns cleaned system-call names.
    Tries both header=None and header='infer' for robustness.
    """
    errors = []
    for header in (None, "infer"):
        try:
            df = pd.read_csv(filepath, header=header)
            if df.empty:
                continue
            first_col = df.columns[0]
            syscalls = (
                df[first_col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            syscalls = [s for s in syscalls if s and s.lower() != "nan"]
            if syscalls:
                return syscalls
        except Exception as e:
            errors.append(str(e))
    raise ValueError(f"Could not parse {filepath}. Details: {' | '.join(errors)}")


def load_sequences_from_folder(folder_path: str, label: int) -> Tuple[List[str], List[int]]:
    sequences, labels = [], []
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not csv_files:
        print(f"[WARNING] No CSV files found in: {folder_path}")
        return sequences, labels

    for filepath in csv_files:
        try:
            syscalls = read_syscalls_from_csv(filepath)

            if len(syscalls) < MIN_SYSCALLS:
                print(f"[SKIP] {os.path.basename(filepath)} has only {len(syscalls)} system calls")
                continue

            early_syscalls = syscalls[:EARLY_STAGE_CALLS]
            sequence = " ".join(early_syscalls)

            sequences.append(sequence)
            labels.append(label)

        except Exception as e:
            print(f"[ERROR] Could not read {filepath}: {e}")

    print(f"Loaded {len(sequences)} samples from '{folder_path}'")
    return sequences, labels


def build_dataset() -> Tuple[List[str], List[int]]:
    print("\n[1/6] Loading dataset...")
    r_seqs, r_labels = load_sequences_from_folder(RANSOMWARE_DIR, label=1)
    b_seqs, b_labels = load_sequences_from_folder(BENIGN_DIR, label=0)

    all_sequences = r_seqs + b_seqs
    all_labels = r_labels + b_labels

    if not all_sequences:
        raise ValueError(
            "Dataset is empty. Make sure CSV files exist in:\n"
            f"  {RANSOMWARE_DIR}\n"
            f"  {BENIGN_DIR}"
        )

    print(f"Total samples  : {len(all_sequences)}")
    print(f"Ransomware (1) : {sum(all_labels)}")
    print(f"Benign     (0) : {all_labels.count(0)}")
    return all_sequences, all_labels


# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(NGRAM_MIN, NGRAM_MAX),
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def get_model_candidates():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "naive_bayes": MultinomialNB(),
        "svm": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    trained_model = clone(model)
    trained_model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)

    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }

    print(f"\n── {name} ──")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Ransomware"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  [Benign]     TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  [Ransomware] FN={cm[1, 0]}  TP={cm[1, 1]}")

    return trained_model, metrics


def compute_cv_f1(name, model, sequences, labels):
    y = np.array(labels)

    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
    folds = max(2, min(CV_FOLDS, min_class_count))

    pipeline = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", clone(model)),
    ])

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, sequences, y, cv=cv, scoring="f1")

    print(f"CV F1 ({folds}-fold): {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean()), float(scores.std())


# ─────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Ransomware Detection - Improved Model Training")
    print("=" * 60)

    # Load data
    sequences, labels = build_dataset()
    y = np.array(labels)

    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        if count < 2:
            raise ValueError(f"Class {cls} has only {count} sample(s). At least 2 are required.")

    # Split raw text first to avoid TF-IDF leakage
    print("\n[2/6] Splitting raw sequences before vectorization...")
    test_size = TEST_SIZE if len(sequences) >= 10 else 0.3
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        sequences,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train samples: {len(X_train_text)} | Test samples: {len(X_test_text)}")

    # Vectorize using training data only
    print("\n[3/6] Fitting TF-IDF on training data only...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Train feature matrix: {X_train.shape}")
    print(f"Test feature matrix : {X_test.shape}")

    # Train and compare models
    print("\n[4/6] Training and evaluating models...")
    models = get_model_candidates()

    trained_models = {}
    results = []
    reports = []

    for name, model in models.items():
        cv_mean, cv_std = compute_cv_f1(name, model, X_train_text, y_train.tolist())
        trained_model, metrics = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        metrics["cv_f1_mean"] = cv_mean
        metrics["cv_f1_std"] = cv_std

        trained_models[name] = trained_model
        results.append(metrics)
        reports.append(
            {
                "model_name": name,
                "classification_report": classification_report(
                    y_test,
                    trained_model.predict(X_test),
                    target_names=["Benign", "Ransomware"],
                    zero_division=0,
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["f1", "recall", "precision", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    print("\n[5/6] Model comparison summary:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    # Save all models + shared vectorizer
    print("\n[6/6] Saving artifacts...")
    save_pickle(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    for name, model in trained_models.items():
        save_pickle(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)

    reports_path = os.path.join(MODEL_DIR, "classification_reports.txt")
    with open(reports_path, "w", encoding="utf-8") as f:
        for report in reports:
            f.write(f"=== {report['model_name']} ===\n")
            f.write(report["classification_report"])
            f.write("\n\n")

    metadata = {
        "best_model": best_model_name,
        "best_model_filename": f"{best_model_name}.pkl",
        "early_stage_calls": EARLY_STAGE_CALLS,
        "min_syscalls": MIN_SYSCALLS,
        "ngram_range": [NGRAM_MIN, NGRAM_MAX],
        "max_features": MAX_FEATURES,
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "labels": {"0": "Benign", "1": "Ransomware"},
    }
    save_json(metadata, os.path.join(MODEL_DIR, "metadata.json"))

    print("\nSaved files:")
    print(f"- {os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')}")
    print(f"- {os.path.join(MODEL_DIR, 'model_results.csv')}")
    print(f"- {os.path.join(MODEL_DIR, 'classification_reports.txt')}")
    print(f"- {os.path.join(MODEL_DIR, 'metadata.json')}")
    for name in trained_models:
        print(f"- {os.path.join(MODEL_DIR, f'{name}.pkl')}")

    print("\nTraining complete.")
    print("Run the dashboard with: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
