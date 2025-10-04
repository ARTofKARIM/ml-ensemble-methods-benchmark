"""Tests for ensemble models."""
import unittest
import numpy as np
from sklearn.datasets import make_classification
from src.models import EnsembleFactory

class TestEnsembleFactory(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=10, random_state=42)

    def test_create_bagging(self):
        model = EnsembleFactory.create_bagging({"n_estimators": 10})
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)

    def test_create_random_forest(self):
        model = EnsembleFactory.create_random_forest({"n_estimators": 10})
        model.fit(self.X, self.y)
        self.assertGreater(model.score(self.X, self.y), 0.5)

    def test_create_all(self):
        models = EnsembleFactory.create_all({"bagging": {"n_estimators": 5}, "random_forest": {"n_estimators": 5},
            "adaboost": {"n_estimators": 5}, "gradient_boosting": {"n_estimators": 5},
            "xgboost": {"n_estimators": 5}, "lightgbm": {"n_estimators": 5}, "stacking": {"cv": 2}})
        self.assertGreater(len(models), 5)

if __name__ == "__main__":
    unittest.main()
