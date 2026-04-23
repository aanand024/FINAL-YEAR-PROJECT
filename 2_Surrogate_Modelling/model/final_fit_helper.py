'''
This file contains a utility function used in the training of the model. 
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import joblib

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


ParamGrid = Union[Dict[str, Any], List[Dict[str, Any]]]


def final_fit_and_save(
    X: np.ndarray,
    y: np.ndarray,
    preprocessor: BaseEstimator,
    model: BaseEstimator,
    param_grid: ParamGrid,
    out_prefix: str = "final_model",
    keep_holdout: bool = False,
    holdout_size: float = 0.10,
    cv_splits: int = 10,
    scoring: str = "r2",
    n_jobs: int = 4,
    random_state: int = 42,
    verbose: int = 1,
) -> Tuple[GridSearchCV, list[str]]:

    # 1) Choose training set (full data or holdout split)
    if keep_holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=holdout_size,
            shuffle=True,
            random_state=random_state,
        )
    else:
        X_train, y_train = X, y
        X_test = y_test = None

    # 2) Build pipeline
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("fs", SelectKBest(score_func=f_regression)),
        ("model", model),
    ])

    # 3) CV for final tuning (not nested as final fit step)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,   # refit best on all of X_train
        n_jobs=n_jobs,
        verbose=verbose,
    )
    gs.fit(X_train, y_train)

    print("\n=== FINAL FIT DONE ===")
    print("Best CV score:", gs.best_score_)
    print("Best params:", gs.best_params_)

    # 4) Extract selected features
    support_mask = gs.best_estimator_.named_steps["fs"].get_support()

    feat_names = None
    prep = gs.best_estimator_.named_steps["prep"]
    if hasattr(prep, "get_feature_names_out"):
        try:
            feat_names = prep.get_feature_names_out()
        except Exception:
            feat_names = None

    if feat_names is not None:
        selected_names = np.array(feat_names)[support_mask].tolist()
    else:
        selected_names = [f"feat_{i}" for i, keep in enumerate(support_mask) if keep]

    feat_path = f"{out_prefix}_selected_features.txt"
    with open(feat_path, "w") as f:
        for name in selected_names:
            f.write(name + "\n")
    print(f"Saved selected features -> {feat_path} (count={len(selected_names)})")

    # 5) Evaluate once on holdout
    if keep_holdout:
        y_pred = gs.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("\nHoldout metrics:")
        print(f"R2={r2:.4f} | MSE={mse:.4f} | MAE={mae:.4f}")

    # 6) Save trained pipeline
    model_path = f"{out_prefix}_pipeline.joblib"
    joblib.dump(gs.best_estimator_, model_path)
    print(f"Saved trained pipeline -> {model_path}")

    return gs, selected_names