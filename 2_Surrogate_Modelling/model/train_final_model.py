from final_fit_helper import final_fit_and_save
from evaluation_helpers import load_and_build_raw_features, make_preprocessor
from xgboost import XGBRegressor

RANDOM_STATE = 42

'''
This file is responsible for the final stage of the surrogate modelling pipeline. 
It trains the chosen model (XGBoost) on the full dataset, ready to be used in the GA. 
'''

if __name__ == "__main__":
    CSV_PATH = "2_Surrogate_Modelling/data/LATEST_all_data_merged_for_regression.csv"
    df, X_raw, y, emb1_slice, emb2_slice, scalar_slice = load_and_build_raw_features(CSV_PATH)
    preprocessor = make_preprocessor(emb1_slice, emb2_slice, scalar_slice)

    # Using best params
    param_grid = {
        "fs__k": [80],  
        "model__max_depth": [3],
        "model__learning_rate": [0.1],
        "model__subsample": [1.0],
        "model__colsample_bytree": [0.8],
        "model__reg_alpha": [0.1],
        "model__reg_lambda": [1.0],
    }


    gs, selected_names = final_fit_and_save(
        X_raw, y,
        preprocessor=preprocessor,
        model=XGBRegressor(random_state=RANDOM_STATE, n_estimators=500),
        param_grid=param_grid,
        out_prefix="final_xgb_full_data",
        keep_holdout=False,  # train on 100%
        n_jobs=4,
        random_state=RANDOM_STATE,
    )