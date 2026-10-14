import pandas as pd
import json
import warnings

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

df = pd.read_csv('athlete_events.csv')
print("Initial dataset size:", df.shape)

df['MedalBinary'] = df['Medal'].notnull().astype(int)
df.dropna(subset=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event'], inplace=True)
print("After dropping missing values:", df.shape)

df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
df['Event_encoded'] = LabelEncoder().fit_transform(df['Event'])

features = ['Age', 'Height', 'Weight', 'BMI', 'Event_encoded', 'Sex_encoded']
X = df[features]
y = df['MedalBinary']

X_smote, y_smote = SMOTE(random_state=42).fit_resample(X, y)
X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X, y)

resampling_sets = {
    'SMOTE': (X_smote, y_smote),
    'Undersampling': (X_under, y_under)
}

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

for resample_name, (X_res, y_res) in resampling_sets.items():
    print(f"\n=== Resampling Strategy: {resample_name} ===")

    print("Training Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    logreg_search = RandomizedSearchCV(
        logreg,
        param_distributions=param_grids['LogisticRegression'],
        cv=3,
        n_iter=10,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    logreg_search.fit(X_res, y_res)
    best_params[f'LogisticRegression_{resample_name}'] = logreg_search.best_params_
    print(f"Best Logistic Regression ({resample_name}):", logreg_search.best_params_)

    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_search = RandomizedSearchCV(
        rf,
        param_distributions=param_grids['RandomForest'],
        cv=3,
        n_iter=10,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_res, y_res)
    best_params[f'RandomForest_{resample_name}'] = rf_search.best_params_
    print(f"Best Random Forest ({resample_name}):", rf_search.best_params_)

    print("Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grids['XGBoost'],
        cv=3,
        n_iter=10,
        scoring='f1',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_res, y_res)
    best_params[f'XGBoost_{resample_name}'] = xgb_search.best_params_
    print(f"Best XGBoost ({resample_name}):", xgb_search.best_params_)

# Save
with open("best_model_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

print("\nAll best parameters saved to best_model_params.json")
