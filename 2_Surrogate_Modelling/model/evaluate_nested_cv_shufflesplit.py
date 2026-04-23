'''
This file is responsible for Stage 1 of the surrogate modelling pipeline. 
It contains the code used in the to compare different modelling families.  
'''

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone as sk_clone
from sklearn.model_selection import (
    KFold,
    train_test_split,
    GridSearchCV,
    ShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1) DATA LOADING + RAW FEATURE BUILDING
# ============================================================
def load_and_build_raw_features(csv_path: str):
    df = pd.read_csv(csv_path)

    df["Embedding1_Neutral"] = df["Embedding1_Neutral"].apply(lambda x: np.array(eval(x)).flatten())
    df["Embedding2_Neutral"] = df["Embedding2_Neutral"].apply(lambda x: np.array(eval(x)).flatten())

    emb1 = np.stack(df["Embedding1_Neutral"].values)  # (n, d1)
    emb2 = np.stack(df["Embedding2_Neutral"].values)  # (n, d2)
    d1, d2 = emb1.shape[1], emb2.shape[1]

    # Engineered scalar features
    emb1_norm = np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = np.linalg.norm(emb2, axis=1, keepdims=True)

    emb_mean1 = emb1.mean(axis=1, keepdims=True)
    emb_mean2 = emb2.mean(axis=1, keepdims=True)
    emb_std1 = emb1.std(axis=1, keepdims=True)
    emb_std2 = emb2.std(axis=1, keepdims=True)
    emb_max1 = emb1.max(axis=1, keepdims=True)
    emb_max2 = emb2.max(axis=1, keepdims=True)
    emb_min1 = emb1.min(axis=1, keepdims=True)
    emb_min2 = emb2.min(axis=1, keepdims=True)

    diff_cos = df[["difference_cos1", "difference_cos2"]].values

    # Raw features: [emb1_raw, emb2_raw, engineered scalars]
    X_raw = np.hstack(
        [
            emb1,
            emb2,
            emb1_norm,
            emb2_norm,
            emb_mean1,
            emb_mean2,
            emb_std1,
            emb_std2,
            emb_max1,
            emb_max2,
            emb_min1,
            emb_min2,
            diff_cos,
        ]
    ).astype(np.float32)

    y = df["prompt_bias_score"].values.astype(np.float32)

    # Slices for ColumnTransformer
    emb1_slice = slice(0, d1)
    emb2_slice = slice(d1, d1 + d2)
    scalar_slice = slice(d1 + d2, X_raw.shape[1])

    return df, X_raw, y, emb1_slice, emb2_slice, scalar_slice


# ============================================================
# 2) PREPROCESSOR (PCA FITS INSIDE CV ONLY)
# ============================================================
def make_preprocessor(emb1_slice, emb2_slice, scalar_slice):
    emb1_pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=30, random_state=RANDOM_STATE)),
        ]
    )
    emb2_pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=30, random_state=RANDOM_STATE)),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("emb1", emb1_pipe, emb1_slice),
            ("emb2", emb2_pipe, emb2_slice),
            ("scalars", "passthrough", scalar_slice),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    prep = Pipeline(
        steps=[
            ("ct", ct),
            ("final_scale", StandardScaler()),
        ]
    )
    return prep


# ============================================================
# 3) MODELS + PARAM GRIDS
# ============================================================
def get_param_grids():
    common = {
        "prep__ct__emb1__pca__n_components": [30],   # fixed for now
        "prep__ct__emb2__pca__n_components": [30],   
        "fs__score_func": [f_regression],            
        "fs__k": [40],                               
    }

    grids = []

    grids.append({**common, "model": [LinearRegression()]})

    grids.append({
        **common,
        "model": [Ridge(random_state=RANDOM_STATE)],
        "model__alpha": [0.1, 1.0, 10.0],
    })

    grids.append({
        **common,
        "model": [ElasticNet(random_state=RANDOM_STATE, max_iter=5000)],
        "model__alpha": [0.1, 1.0],
        "model__l1_ratio": [0.5],
    })

    grids.append({
        **common,
        "model": [SVR()],
        "model__C": [1.0, 5.0],
        "model__epsilon": [0.1],
        "model__kernel": ["rbf"],
    })

    grids.append({
        **common,
        "model": [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=4)],
        "model__n_estimators": [300],
        "model__max_depth": [None, 6],
    })

    grids.append({
        **common,
        "model": [GradientBoostingRegressor(random_state=RANDOM_STATE)],
        "model__n_estimators": [300],
        "model__learning_rate": [0.05],
    })

    grids.append({
        **common,
        "model": [LGBMRegressor(random_state=RANDOM_STATE, n_jobs=4, verbose=-1)],
        "model__n_estimators": [300],
        "model__learning_rate": [0.05],
        "model__num_leaves": [31],
    })

    grids.append({
        **common,
        "model": [XGBRegressor(random_state=RANDOM_STATE, n_jobs=4, tree_method="hist")],
        "model__n_estimators": [300],
        "model__learning_rate": [0.05],
        "model__max_depth": [3],
    })

    grids.append({
        **common,
        "model": [CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)],
        "model__iterations": [300],
        "model__learning_rate": [0.05],
        "model__depth": [6],
    })

    return grids


# ============================================================
# 4) FEATURE NAMES AFTER PREPROCESSING + SELECTION
# ============================================================
def get_selected_feature_names(fitted_pipeline: Pipeline):
    prep = fitted_pipeline.named_steps["prep"]
    fs = fitted_pipeline.named_steps["fs"]

    names = prep.named_steps["ct"].get_feature_names_out()
    mask = fs.get_support()
    selected = np.array(names)[mask]
    return selected.tolist()


# ============================================================
# 5) Fit GridSearch on the 80% training chunk (inside the 90% pool)
# ============================================================
def fit_grid_on_train80(X_train80, y_train80, preprocessor, param_grids, rng):
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("fs", SelectKBest(score_func=f_regression, k=40)),
            ("model", Ridge(random_state=RANDOM_STATE)),  # placeholder -> grid overwrites
        ]
    )

    inner_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=int(rng.randint(0, 1_000_000)),
    )

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids,
        scoring="r2",
        cv=inner_cv,
        refit=True,  # refit best on ALL of train 80
        n_jobs=4,
        verbose=0,
    )
    gs.fit(X_train80, y_train80)
    return gs


# ============================================================
# 6) WORKFLOW:
#    10x (90/10 split) × [5x ShuffleSplit inside 90%: train 80/val 10]
# ============================================================
def run_workflow(
    X_raw,
    y,
    emb1_slice,
    emb2_slice,
    scalar_slice,
    n_repeats=10,
    n_val_splits=5,
    csv_out_prefix="nestedcv",
):
    param_grids = get_param_grids()

    all_repeat_final = []
    all_val_rows = []
    all_repeat_selected_features = []
    fold_records = [] 

    for rep in range(1, n_repeats + 1):
        # 90/10 split (final held-out test)
        X_90, X_10, y_90, y_10 = train_test_split(
            X_raw,
            y,
            test_size=0.10,
            random_state=RANDOM_STATE + rep,
            shuffle=True,
        )

        preprocessor = make_preprocessor(emb1_slice, emb2_slice, scalar_slice)

        # inside the 90% pool: validation = 10% of ORIGINAL => 1/9 of 90%
        ss = ShuffleSplit(
            n_splits=n_val_splits,
            test_size=1/9,
            random_state=RANDOM_STATE + 1000 + rep,
        )

        split_scores = []
        split_candidates = []  # (val_r2, best_params, best_model_name)

        for split_i, (tr_idx, val_idx) in enumerate(ss.split(X_90), start=1):
            X_train80, y_train80 = X_90[tr_idx], y_90[tr_idx]
            X_val10, y_val10 = X_90[val_idx], y_90[val_idx]

            rng_split = np.random.RandomState(RANDOM_STATE + rep * 10_000 + split_i)

            gs = fit_grid_on_train80(X_train80, y_train80, preprocessor, param_grids, rng_split)

            best_est = gs.best_estimator_
            y_hat = best_est.predict(X_val10)

            val_r2 = float(r2_score(y_val10, y_hat))
            val_mse = float(mean_squared_error(y_val10, y_hat))
            val_mae = float(mean_absolute_error(y_val10, y_hat))
            best_model_name = best_est.named_steps["model"].__class__.__name__

            fold_records.append({
                "repeat": rep,
                "split": split_i,
                "model": best_model_name,
                "val_R2": float(val_r2),
                "val_MSE": float(val_mse),
                "val_MAE": float(val_mae),
                "inner_best_R2": float(gs.best_score_),
            })

            split_scores.append(val_r2)
            split_candidates.append((val_r2, gs.best_params_, best_model_name))

            all_val_rows.append({
                "repeat": rep,
                "split": split_i,
                "inner_cv_best_r2": float(gs.best_score_),
                "val_r2": val_r2,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "best_model": best_model_name,
                "best_params": str(gs.best_params_),
            })

            print(
                f"[Repeat {rep:02d} | Split {split_i}] "
                f"inner_best_R2={gs.best_score_:.4f} | val_R2={val_r2:.4f} | model={best_model_name}"
            )

        mean_val_r2 = float(np.mean(split_scores))

        # Choose best params for this repeat (highest val R2)
        best_idx = int(np.argmax(np.array(split_scores)))
        chosen_val_r2, chosen_params, _ = split_candidates[best_idx]

        # Refit chosen config on FULL 90%
        chosen_params = dict(chosen_params)
        if "model" in chosen_params:
            chosen_params["model"] = sk_clone(chosen_params["model"])

        final_pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("fs", SelectKBest(score_func=f_regression, k=40)),
                ("model", Ridge(random_state=RANDOM_STATE)),  # placeholder
            ]
        )
        final_pipe.set_params(**chosen_params)
        final_pipe.fit(X_90, y_90)

        # Test on held-out 10%
        y10_hat = final_pipe.predict(X_10)
        test_r2 = float(r2_score(y_10, y10_hat))
        test_mse = float(mean_squared_error(y_10, y10_hat))
        test_mae = float(mean_absolute_error(y_10, y10_hat))

        selected_features = get_selected_feature_names(final_pipe)
        all_repeat_selected_features.append({
            "repeat": rep,
            "chosen_model": final_pipe.named_steps["model"].__class__.__name__,
            "chosen_params": str(chosen_params),
            "n_selected_features": len(selected_features),
            "selected_features": ";".join(selected_features),
        })

        all_repeat_final.append({
            "repeat": rep,
            "mean_val_r2_on_90": mean_val_r2,
            "chosen_val_r2_on_90": float(chosen_val_r2),
            "final_test_r2_on_10": test_r2,
            "final_test_mse_on_10": test_mse,
            "final_test_mae_on_10": test_mae,
            "chosen_model": final_pipe.named_steps["model"].__class__.__name__,
        })

        print(
            f"=== Repeat {rep:02d} DONE | mean_val_R2(on 90)={mean_val_r2:.4f} | "
            f"chosen_val_R2(on 90)={chosen_val_r2:.4f} | test_R2(on 10)={test_r2:.4f} | "
            f"chosen={final_pipe.named_steps['model'].__class__.__name__}\n"
        )

    df_val = pd.DataFrame(all_val_rows)
    df_final = pd.DataFrame(all_repeat_final)
    df_feats = pd.DataFrame(all_repeat_selected_features)

    df_val.to_csv(f"{csv_out_prefix}_val_splits.csv", index=False)
    df_final.to_csv(f"{csv_out_prefix}_final_test.csv", index=False)
    df_feats.to_csv(f"{csv_out_prefix}_selected_features.csv", index=False)

    print("===============================================")
    print("FINAL SUMMARY ACROSS REPEATS")
    print("===============================================")
    print(f"Final test R2 mean ± std: {df_final['final_test_r2_on_10'].mean():.4f} ± {df_final['final_test_r2_on_10'].std():.4f}")
    print(f"Final test MSE mean ± std: {df_final['final_test_mse_on_10'].mean():.4f} ± {df_final['final_test_mse_on_10'].std():.4f}")
    print(f"Final test MAE mean ± std: {df_final['final_test_mae_on_10'].mean():.4f} ± {df_final['final_test_mae_on_10'].std():.4f}")
    print("\nChosen model counts:")
    print(df_final["chosen_model"].value_counts())

    print(f"\nSaved:\n- {csv_out_prefix}_val_splits.csv\n- {csv_out_prefix}_final_test.csv\n- {csv_out_prefix}_selected_features.csv")

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(f"{csv_out_prefix}_shufflesplit_fold_level_results.csv", index=False)
    print("Saved fold-level results.")

    return df_val, df_final, df_feats


if __name__ == "__main__":
    CSV_PATH = "2_Surrogate_Modelling/data/LATEST_all_data_merged_for_regression.csv"

    df, X_raw, y, emb1_slice, emb2_slice, scalar_slice = load_and_build_raw_features(CSV_PATH)

    print(f"X_raw shape: {X_raw.shape}")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

    run_workflow(
        X_raw=X_raw,
        y=y,
        emb1_slice=emb1_slice,
        emb2_slice=emb2_slice,
        scalar_slice=scalar_slice,
        n_repeats=10,
        n_val_splits=5,
        csv_out_prefix="nestedcv_surrogate",
    )
