"""Evaluation and benchmarking module."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

class Benchmarker:
    def __init__(self):
        self.results = {}

    def benchmark(self, models, X_train, X_test, y_train, y_test, cv_folds=5):
        for name, model in models.items():
            print(f"Training {name}...")
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
            self.results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "cv_accuracy": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "train_time": train_time,
            }
            print(f"  {name}: acc={self.results[name]['accuracy']:.4f}, f1={self.results[name]['f1_macro']:.4f}, time={train_time:.2f}s")
        return self.results

    def comparison_table(self):
        rows = []
        for name, m in self.results.items():
            rows.append({"Model": name, "Accuracy": f"{m['accuracy']:.4f}", "F1": f"{m['f1_macro']:.4f}",
                         "CV Acc": f"{m['cv_accuracy']:.4f}±{m['cv_std']:.4f}", "Time (s)": f"{m['train_time']:.2f}"})
        return pd.DataFrame(rows)

    def best_model(self, metric="accuracy"):
        return max(self.results, key=lambda k: self.results[k][metric])
