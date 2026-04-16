import warnings
from sklearn.base import clone as sk_clone
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    train_test_split,
    PredefinedSplit,
    GridSearchCV,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, ShuffleSplit

'''
This file contains helper functions used in the evaluation of models. 
'''

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1) DATA LOADING + RAW FEATURE BUILDING (NO PCA HERE)
# ============================================================
def load_and_build_raw_features(csv_path: str):
    df = pd.read_csv(csv_path)

    df["Embedding1_Neutral"] = df["Embedding1_Neutral"].apply(
        lambda x: np.array(eval(x)).flatten()
    )
    df["Embedding2_Neutral"] = df["Embedding2_Neutral"].apply(
        lambda x: np.array(eval(x)).flatten()
    )

    emb1 = np.stack(df["Embedding1_Neutral"].values)  # (n, d1)
    emb2 = np.stack(df["Embedding2_Neutral"].values)  # (n, d2)
    d1, d2 = emb1.shape[1], emb2.shape[1]

    # Engineered scalar features (safe to compute upfront; no fitting)
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