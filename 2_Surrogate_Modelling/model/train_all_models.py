'''
This file is responsible for Stage 2 of the surrogate modelling pipeline. 
It completes the final tuning and evaluation for all considered models. 
'''

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from final_fit_helper import final_fit_and_save
from evaluation_helpers import load_and_build_raw_features, make_preprocessor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

if __name__ == "__main__":
    CSV_PATH = "2_Surrogate_Modelling/data/LATEST_all_data_merged_for_regression.csv"

    df, X_raw, y, emb1_slice, emb2_slice, scalar_slice = load_and_build_raw_features(CSV_PATH)
    preprocessor = make_preprocessor(emb1_slice, emb2_slice, scalar_slice)

    model_configs = [
        {
            "name": "ElasticNet",
            "model": ElasticNet(max_iter=20000, random_state=RANDOM_STATE),
            "param_grid": {
                "fs__k": [20, 40, 60, 80],
                "model__alpha": [1e-3, 1e-2, 1e-1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        },
        {
            "name": "Ridge",
            "model": Ridge(random_state=RANDOM_STATE),
            "param_grid": {
                "fs__k": [20, 40, 60, 80],
                "model__alpha": [0.1, 1.0, 10.0],
            },
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
        {
            "name": "LGBM",
            "model": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [31, 63],
                "model__max_depth": [3, 5, 7],
            },
        },
        {
            "name": "CatBoost",
            "model": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__iterations": [200, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__depth": [3, 5, 7],
            },
        },
        {
            "name": "XGB",
            "model": XGBRegressor(random_state=RANDOM_STATE, n_estimators=500),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_alpha": [0.0, 0.1],
                "model__reg_lambda": [1.0, 5.0],
            },
        },
        {
            "name": "SVR",
            "model": SVR(),
            "param_grid": {
                "fs__k": [40, 80, 120],
                "model__C": [0.1, 1, 10],
                "model__epsilon": [0.01, 0.1, 0.2],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        },
    ]

    all_results = []
    for config in model_configs:
        print(f"\n=== Fitting {config['name']} ===")
        gs, selected_names = final_fit_and_save(
            X_raw, y,
            preprocessor=preprocessor,
            model=config["model"],
            param_grid=config["param_grid"],
            out_prefix=f"w_x_final_surrogate_{config['name'].lower()}",
            keep_holdout=True,   # validate on 10, train on 90
            n_jobs=4,
            random_state=RANDOM_STATE,
        )

        # Extract holdout metrics
        # The holdout split is done inside final_fit_and_save, so we repeat it here to get the same split
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y,
            test_size=0.10,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        y_pred = gs.predict(X_test)
        holdout_r2 = r2_score(y_test, y_pred)
        holdout_mse = mean_squared_error(y_test, y_pred)
        holdout_mae = mean_absolute_error(y_test, y_pred)

        all_results.append({
            "model": config["name"],
            "best_score": gs.best_score_,
            "best_params": gs.best_params_,
            "selected_features": ";".join(selected_names),
            "holdout_r2": holdout_r2,
            "holdout_mse": holdout_mse,
            "holdout_mae": holdout_mae,
        })

    pd.DataFrame(all_results).to_csv("final_surrogate_all_models_summary.csv", index=False)