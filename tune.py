import json
import warnings

from sklearn.model_selection import RandomizedSearchCV

from common import load_data, split_train_test, make_pipeline, FEATURES, RANDOM_STATE

warnings.filterwarnings("ignore")

df = load_data()

# tune on pre-2016 Games only so the 2016 test set stays unseen
train_df, _ = split_train_test(df)
X = train_df[FEATURES]
y = train_df['MedalBinary']

param_grids = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

best_params = {}

for resample_name in ['SMOTE', 'Undersampling']:
    print(f"\n=== Resampling Strategy: {resample_name} ===")

    for model_name, grid in param_grids.items():
        print(f"Training {model_name}...")
        # resampling happens inside each CV fold; average precision is
        # threshold-independent since main.py tunes the threshold separately
        search = RandomizedSearchCV(
            make_pipeline(model_name, resample_name, n_jobs=1),
            param_distributions={f'clf__{k}': v for k, v in grid.items()},
            cv=3,
            n_iter=10,
            scoring='average_precision',
            verbose=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        search.fit(X, y)
        params = {k.removeprefix('clf__'): v for k, v in search.best_params_.items()}
        best_params[f'{model_name}_{resample_name}'] = params
        print(f"Best {model_name} ({resample_name}):", params, flush=True)

# Save
with open("best_model_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("\nAll best parameters saved to best_model_params.json")
