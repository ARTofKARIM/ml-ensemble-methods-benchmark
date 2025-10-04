"""Visualization for ensemble benchmark."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class BenchmarkVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_comparison(self, results, metric="accuracy", save=True):
        models = list(results.keys())
        values = [results[m][metric] for m in models]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Ensemble Methods Comparison - {metric}")
        ax.set_ylim(min(values) * 0.95, max(values) * 1.02)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        if save:
            fig.savefig(f"{self.output_dir}{metric}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_time_comparison(self, results, save=True):
        models = list(results.keys())
        times = [results[m]["train_time"] for m in models]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(models, times, color="steelblue")
        ax.set_xlabel("Training Time (s)")
        ax.set_title("Training Time Comparison")
        if save:
            fig.savefig(f"{self.output_dir}time_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
