"""Main entry point for ensemble benchmark."""
import argparse
import yaml
from src.data_loader import DataLoader
from src.models import EnsembleFactory
from src.evaluation import Benchmarker
from src.visualization import BenchmarkVisualizer

def main():
    parser = argparse.ArgumentParser(description="Ensemble Methods Benchmark")
    parser.add_argument("--data", help="Dataset path")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(args.config)
    df = loader.load(args.data)
    df = loader.preprocess(df)
    X_train, X_test, y_train, y_test = loader.split(df)

    models = EnsembleFactory.create_all(config["models"])
    bench = Benchmarker()
    results = bench.benchmark(models, X_train, X_test, y_train, y_test, config["evaluation"]["cv_folds"])

    print("\n" + "=" * 80)
    print(bench.comparison_table().to_string(index=False))
    print(f"\nBest model: {bench.best_model()}")

    viz = BenchmarkVisualizer()
    viz.plot_comparison(results, "accuracy")
    viz.plot_comparison(results, "f1_macro")
    viz.plot_time_comparison(results)

if __name__ == "__main__":
    main()
