from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import pandas as pd

from .schemas import ExecutionConfig


@dataclass
class ModelHandle:
    backend_name: str
    model: Any
    predictors: list[str]
    target: str
    model_id: str
    training_seconds: float
    execution_config: ExecutionConfig


class BaseBackend:
    name = "base"

    def fit_automl(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        predictors: list[str],
        config: ExecutionConfig,
    ) -> ModelHandle:
        raise NotImplementedError

    def fit_manual_baseline(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        predictors: list[str],
        seed: int,
    ) -> ModelHandle:
        raise NotImplementedError

    def predict_proba(self, handle: ModelHandle, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def global_local_explanations(
        self, handle: ModelHandle, df: pd.DataFrame, local_row: int = 0
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        return None, None

    def cleanup(self) -> None:
        """Release backend-side temporary objects between independent runs."""
        return None


class H2OBackend(BaseBackend):
    name = "h2o"

    def __init__(self, port: int = 54321, max_mem_size: str = "4G"):
        try:
            import h2o
            from h2o.automl import H2OAutoML
            from h2o.estimators.gbm import H2OGradientBoostingEstimator
            from h2o.estimators.xgboost import H2OXGBoostEstimator
        except ImportError as exc:
            raise ImportError(
                "H2O is required for manuscript experiments. Install requirements.txt first."
            ) from exc
        self.h2o = h2o
        self.H2OAutoML = H2OAutoML
        self.H2OGradientBoostingEstimator = H2OGradientBoostingEstimator
        self.H2OXGBoostEstimator = H2OXGBoostEstimator
        h2o.init(port=port, max_mem_size=max_mem_size)
        # H2O-3 does not expose XGBoost on Windows. Query the running server
        # instead of assuming that every client-side estimator is executable.
        try:
            self.xgboost_available = bool(H2OXGBoostEstimator.available())
        except Exception:
            self.xgboost_available = False
        self.supported_automl_algos = {"GBM", "DRF", "GLM", "StackedEnsemble"}
        if self.xgboost_available:
            self.supported_automl_algos.add("XGBoost")

    def cleanup(self) -> None:
        """Clear H2O's distributed key-value store without stopping the cluster.

        Each condition-method run is independent after its metrics have been
        materialized into the CSV row. Clearing models/frames here prevents
        hundreds of AutoML runs from accumulating in the 4 GB JVM heap.
        """
        import gc

        try:
            self.h2o.remove_all()
        finally:
            gc.collect()

    def _frames(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        predictors: list[str],
    ):
        keep = predictors + [target]
        train_h2o = self.h2o.H2OFrame(train[keep])
        valid_h2o = self.h2o.H2OFrame(validation[keep])
        train_h2o[target] = train_h2o[target].asfactor()
        valid_h2o[target] = valid_h2o[target].asfactor()
        return train_h2o, valid_h2o

    def fit_automl(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        predictors: list[str],
        config: ExecutionConfig,
    ) -> ModelHandle:
        used_predictors = [p for p in predictors if p not in config.drop_features]
        train_h2o, valid_h2o = self._frames(train, validation, target, used_predictors)

        # Keep only algorithms that the active H2O server can actually execute.
        # This is especially important on Windows, where H2O-3 XGBoost is not
        # exposed through the REST API. The effective configuration is logged
        # so manuscript results reflect what was really executed.
        include_algos = [a for a in config.include_algos if a in self.supported_automl_algos]
        if not include_algos:
            include_algos = ["GBM", "DRF", "GLM", "StackedEnsemble"]
        effective_config = config.model_copy(update={"include_algos": include_algos})

        start = time.perf_counter()
        aml = self.H2OAutoML(
            max_models=effective_config.max_models,
            seed=effective_config.seed,
            nfolds=effective_config.nfolds,
            include_algos=effective_config.include_algos,
            balance_classes=effective_config.balance_classes,
            sort_metric=effective_config.sort_metric,
            # Reproducibility is based on a model-count budget, not wall-clock time.
            max_runtime_secs=0,
            verbosity="warn",
        )
        aml.train(
            x=used_predictors,
            y=target,
            training_frame=train_h2o,
            validation_frame=valid_h2o,
        )
        seconds = time.perf_counter() - start
        return ModelHandle(
            backend_name=self.name,
            model=aml.leader,
            predictors=used_predictors,
            target=target,
            model_id=str(aml.leader.model_id),
            training_seconds=seconds,
            execution_config=effective_config,
        )

    def fit_manual_baseline(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        predictors: list[str],
        seed: int,
    ) -> ModelHandle:
        """Fixed manually configured baseline used for ablation B0.

        We use native H2O GBM because H2O-3 XGBoost is unavailable on Windows.
        This still satisfies the reviewer-requested manually configured model
        baseline while keeping the comparison inside the same H2O runtime.
        """
        config = ExecutionConfig(
            include_algos=["GBM"],
            balance_classes=True,
            sort_metric="AUCPR",
            drop_features=[],
            max_models=1,
            seed=seed,
            nfolds=5,
        )
        train_h2o, valid_h2o = self._frames(train, validation, target, predictors)
        start = time.perf_counter()
        model = self.H2OGradientBoostingEstimator(
            ntrees=200,
            max_depth=6,
            learn_rate=0.05,
            sample_rate=0.8,
            col_sample_rate=0.8,
            balance_classes=True,
            seed=seed,
        )
        model.train(
            x=predictors,
            y=target,
            training_frame=train_h2o,
            validation_frame=valid_h2o,
        )
        seconds = time.perf_counter() - start
        return ModelHandle(
            backend_name=self.name,
            model=model,
            predictors=predictors,
            target=target,
            model_id=str(model.model_id),
            training_seconds=seconds,
            execution_config=config,
        )

    def predict_proba(self, handle: ModelHandle, df: pd.DataFrame) -> np.ndarray:
        frame = self.h2o.H2OFrame(df[handle.predictors])
        pred = handle.model.predict(frame).as_data_frame(use_multi_thread=True)
        if "p1" not in pred.columns:
            raise ValueError(f"Expected binary-class probability column 'p1'; got {list(pred.columns)}")
        return pred["p1"].to_numpy(dtype=float)

    def global_local_explanations(
        self, handle: ModelHandle, df: pd.DataFrame, local_row: int = 0
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        try:
            frame = self.h2o.H2OFrame(df[handle.predictors])
            contrib = handle.model.predict_contributions(frame).as_data_frame(use_multi_thread=True)
            feature_cols = [c for c in contrib.columns if c != "BiasTerm"]
            global_imp = (
                contrib[feature_cols].abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
            )
            global_imp.columns = ["feature", "mean_abs_shap"]
            local_row = max(0, min(local_row, len(contrib) - 1))
            local = contrib.iloc[[local_row]].T.reset_index()
            local.columns = ["feature", "shap_contribution"]
            return global_imp, local
        except Exception:
            try:
                varimp = handle.model.varimp(use_pandas=True)
                return varimp, None
            except Exception:
                return None, None


class SklearnSmokeBackend(BaseBackend):
    """Offline backend for unit/smoke tests only; not for manuscript results."""

    name = "sklearn_smoke"

    def _pipeline(self, algo: str, seed: int, balance: bool):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        class_weight = "balanced" if balance else None

        if algo == "GBM" or algo == "XGBoost":
            clf = GradientBoostingClassifier(random_state=seed)
        elif algo == "DRF":
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=seed,
                class_weight=class_weight,
                n_jobs=-1,
            )
        elif algo == "GLM":
            clf = LogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight=class_weight,
            )
        elif algo == "StackedEnsemble":
            clf = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=120, random_state=seed, class_weight=class_weight)),
                    ("lr", LogisticRegression(max_iter=2000, random_state=seed, class_weight=class_weight)),
                ],
                voting="soft",
            )
        else:
            raise ValueError(algo)
        return clf

    def _fit_one(self, train, validation, target, predictors, algo, seed, balance):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        x_train = train[predictors]
        categorical = [c for c in predictors if not pd.api.types.is_numeric_dtype(x_train[c])]
        numeric = [c for c in predictors if c not in categorical]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
            ],
            remainder="drop",
        )
        clf = self._pipeline(algo, seed, balance)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(x_train, train[target].to_numpy(dtype=int))
        return pipe

    def fit_automl(self, train, validation, target, predictors, config):
        from sklearn.metrics import average_precision_score

        used = [p for p in predictors if p not in config.drop_features]
        start = time.perf_counter()
        algos = config.include_algos[: max(1, config.max_models)]
        best = None
        best_score = -np.inf
        best_algo = None
        for i, algo in enumerate(algos):
            try:
                model = self._fit_one(train, validation, target, used, algo, config.seed + i, config.balance_classes)
                prob = model.predict_proba(validation[used])[:, 1]
                score = average_precision_score(validation[target], prob)
                if score > best_score:
                    best_score, best, best_algo = score, model, algo
            except Exception:
                continue
        if best is None:
            raise RuntimeError("No smoke-test candidate model could be fitted.")
        return ModelHandle(
            backend_name=self.name,
            model=best,
            predictors=used,
            target=target,
            model_id=f"sklearn_{best_algo}_{config.seed}",
            training_seconds=time.perf_counter() - start,
            execution_config=config,
        )

    def fit_manual_baseline(self, train, validation, target, predictors, seed):
        config = ExecutionConfig(
            include_algos=["GBM"], balance_classes=True, sort_metric="AUCPR",
            drop_features=[], max_models=1, seed=seed, nfolds=5,
        )
        start = time.perf_counter()
        model = self._fit_one(train, validation, target, predictors, "GBM", seed, True)
        return ModelHandle(
            backend_name=self.name,
            model=model,
            predictors=predictors,
            target=target,
            model_id=f"sklearn_manual_gbm_{seed}",
            training_seconds=time.perf_counter() - start,
            execution_config=config,
        )

    def predict_proba(self, handle: ModelHandle, df: pd.DataFrame) -> np.ndarray:
        return handle.model.predict_proba(df[handle.predictors])[:, 1].astype(float)


def build_backend(name: str, *, h2o_port: int = 54321, h2o_max_mem_size: str = "4G") -> BaseBackend:
    if name == "h2o":
        return H2OBackend(port=h2o_port, max_mem_size=h2o_max_mem_size)
    if name == "sklearn_smoke":
        return SklearnSmokeBackend()
    raise ValueError(f"Unsupported backend {name!r}")
