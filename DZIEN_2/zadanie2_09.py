#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Szkoleniowe Zadanie: Klasyfikacja Iris (Random Forest)

Realizuje kroki:
1) wczytanie danych  2) podział train/test  3) trening RF
4) predykcja         5) ocena (accuracy, raport, macierz)
6) wizualizacja (opcjonalnie: macierz + ważności)
7) zapis wyników do plików

Wymagane: pandas, numpy, scikit-learn, matplotlib, seaborn
Instalacja (opcjonalnie):
    pip install -U pandas numpy scikit-learn matplotlib seaborn
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

RNG_SEED = 42


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_data():
    """Krok 3: wczytanie zestawu danych Iris (cechy + etykiety)."""
    iris = load_iris(as_frame=True)
    X = iris.data                    # DataFrame z 4 cechami
    y = iris.target                  # Series z etykietami 0/1/2
    feature_names = list(X.columns)
    target_names = list(iris.target_names)
    return X, y, feature_names, target_names


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Krok 4: podział na zbiory treningowy i testowy (stratyfikowany)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RNG_SEED,
        stratify=y
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Krok 5: inicjalizacja i trening modelu Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RNG_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_and_save(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str],
    outdir: str = "outputs"
) -> dict:
    """Kroki 6–8: predykcja, ocena, wizualizacja, zapisy do plików."""
    ensure_outdir(outdir)

    # Predykcja
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)
        if hasattr(model, "predict_proba")
        else None
    )

    # Metryki
    acc = accuracy_score(y_test, y_pred)
    clf_rep = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # --- Konsola ---
    print("\n=== WYNIKI ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Raport klasyfikacji:")
    print(clf_rep)
    print("Macierz pomyłek:")
    print(cm)

    # Zapis raportu do pliku
    with open(os.path.join(outdir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(clf_rep)

    # Wykres: macierz pomyłek
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista")
    plt.title("Macierz pomyłek – Random Forest (Iris)")
    plt.tight_layout()
    cm_path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Wykres: ważności cech
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.xlabel("Ważność (feature importance)")
    plt.ylabel("Cecha")
    plt.title("Ważność cech – Random Forest (Iris)")
    plt.tight_layout()
    fi_path = os.path.join(outdir, "feature_importances.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()

    # Zapis predykcji do CSV (z cechami)
    pred_df = X_test.reset_index(drop=True).copy()
    pred_df["y_true"] = pd.Categorical.from_codes(y_test.values, categories=target_names)
    pred_df["y_pred"] = pd.Categorical.from_codes(y_pred, categories=target_names)
    if y_proba is not None:
        proba_df = pd.DataFrame(y_proba, columns=[f"proba_{n}" for n in target_names])
        pred_df = pd.concat([pred_df, proba_df], axis=1)

    csv_path = os.path.join(outdir, "iris_predictions.csv")
    pred_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Dodatkowo zapis metadanych do JSON
    meta = {
        "accuracy": float(acc),
        "target_names": target_names,
        "n_test_samples": int(len(y_test)),
        "outputs": {
            "classification_report_txt": "classification_report.txt",
            "confusion_matrix_png": "confusion_matrix.png",
            "feature_importances_png": "feature_importances.png",
            "predictions_csv": "iris_predictions.csv"
        }
    }
    with open(os.path.join(outdir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"accuracy": acc, "cm_path": cm_path, "fi_path": fi_path, "csv_path": csv_path}


def main():
    print("Krok 1–3: wczytywanie danych Iris…")
    X, y, feature_names, target_names = load_data()
    print(f"Cechy: {feature_names}")
    print(f"Etykiety: {target_names}")
    print(f"Kształt X: {X.shape}, y: {y.shape}")

    print("\nKrok 4: podział train/test (80/20, stratified)…")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nKrok 5: trening modelu Random Forest…")
    model = train_model(X_train, y_train)

    print("\nKroki 6–8: predykcja, ocena, wizualizacja, zapisy…")
    results = evaluate_and_save(model, X_test, y_test, target_names, outdir="outputs")

    print("\nZrobione.")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Pliki w katalogu ./outputs:")
    print(" - classification_report.txt")
    print(" - confusion_matrix.png")
    print(" - feature_importances.png")
    print(" - iris_predictions.csv")
    print(" - run_metadata.json")


if __name__ == "__main__":
    main()
