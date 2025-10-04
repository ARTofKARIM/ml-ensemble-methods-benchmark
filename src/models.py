"""Ensemble model implementations."""
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

class EnsembleFactory:
    @staticmethod
    def create_bagging(config):
        return BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=config.get("n_estimators", 100),
                                  max_samples=config.get("max_samples", 0.8), random_state=42, n_jobs=-1)

    @staticmethod
    def create_random_forest(config):
        return RandomForestClassifier(n_estimators=config.get("n_estimators", 200),
                                       max_depth=config.get("max_depth", 15), random_state=42, n_jobs=-1)

    @staticmethod
    def create_adaboost(config):
        return AdaBoostClassifier(n_estimators=config.get("n_estimators", 100),
                                   learning_rate=config.get("learning_rate", 0.1), random_state=42)

    @staticmethod
    def create_gradient_boosting(config):
        return GradientBoostingClassifier(n_estimators=config.get("n_estimators", 200),
                                           learning_rate=config.get("learning_rate", 0.05),
                                           max_depth=config.get("max_depth", 5), random_state=42)

    @staticmethod
    def create_xgboost(config):
        return xgb.XGBClassifier(n_estimators=config.get("n_estimators", 300),
                                  learning_rate=config.get("learning_rate", 0.03),
                                  use_label_encoder=False, eval_metric="mlogloss", random_state=42)

    @staticmethod
    def create_lightgbm(config):
        return lgb.LGBMClassifier(n_estimators=config.get("n_estimators", 300),
                                   learning_rate=config.get("learning_rate", 0.05), random_state=42, verbose=-1)

    @staticmethod
    def create_stacking(config):
        estimators = [("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                      ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
                      ("xgb", xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42))]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
                                   cv=config.get("cv", 5), n_jobs=-1)

    @staticmethod
    def create_voting():
        estimators = [("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                      ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
                      ("xgb", xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42))]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    @classmethod
    def create_all(cls, config):
        return {
            "Bagging": cls.create_bagging(config.get("bagging", {})),
            "Random Forest": cls.create_random_forest(config.get("random_forest", {})),
            "AdaBoost": cls.create_adaboost(config.get("adaboost", {})),
            "Gradient Boosting": cls.create_gradient_boosting(config.get("gradient_boosting", {})),
            "XGBoost": cls.create_xgboost(config.get("xgboost", {})),
            "LightGBM": cls.create_lightgbm(config.get("lightgbm", {})),
            "Stacking": cls.create_stacking(config.get("stacking", {})),
            "Voting": cls.create_voting(),
        }
