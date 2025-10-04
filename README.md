# Ensemble Methods Benchmark

A comprehensive benchmark comparing 8 ensemble learning methods: Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Stacking, and Voting classifiers.

## Architecture
```
ml-ensemble-methods-benchmark/
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models.py         # Ensemble factory (8 methods)
│   ├── evaluation.py     # Cross-validation benchmarking
│   └── visualization.py  # Comparison charts
├── config/config.yaml
├── tests/test_models.py
└── main.py
```

## Models Benchmarked
| Model | Type | Library |
|-------|------|---------|
| Bagging | Bootstrap aggregating | scikit-learn |
| Random Forest | Bagging + feature randomness | scikit-learn |
| AdaBoost | Adaptive boosting | scikit-learn |
| Gradient Boosting | Sequential boosting | scikit-learn |
| XGBoost | Regularized boosting | xgboost |
| LightGBM | Histogram-based boosting | lightgbm |
| Stacking | Meta-learning | scikit-learn |
| Voting | Soft voting ensemble | scikit-learn |

## Installation
```bash
git clone https://github.com/mouachiqab/ml-ensemble-methods-benchmark.git
cd ml-ensemble-methods-benchmark
pip install -r requirements.txt
```

## Usage
```bash
python main.py --data data/dataset.csv
```

## Technologies
- Python 3.9+, scikit-learn, XGBoost, LightGBM, CatBoost







