import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

# load dataset
df = pd.read_csv('athlete_events.csv')
print("Initial dataset size:", df.shape)

# create binary label
df['MedalBinary'] = df['Medal'].notnull().astype(int)

# drop rows with missing values
df.dropna(subset=['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event'], inplace=True)
print("After dropping missing values:", df.shape)

# create features
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Sex_encoded'] = LabelEncoder().fit_transform(df['Sex'])
df['Event_encoded'] = LabelEncoder().fit_transform(df['Event'])

features = ['Age', 'Height', 'Weight', 'BMI', 'Event_encoded', 'Sex_encoded']
df = df.copy()


# split train/test
train_df = df[df['Year'] < 2016]
test_df = df[df['Year'] == 2016]
train_ids = set(train_df['ID'])
test_df = test_df[~test_df['ID'].isin(train_ids)]

X_train = train_df[features]
y_train = train_df['MedalBinary']
X_test = test_df[features]
y_test = test_df['MedalBinary']



# resampling
X_train_smote, y_train_smote = SMOTE(random_state=15).fit_resample(X_train, y_train)
X_train_under, y_train_under = RandomUnderSampler(random_state=15).fit_resample(X_train, y_train)



# plot class distributions
fig, axes = plt.subplots(3, 1, figsize=(6, 10))
sns.countplot(x=y_train, ax=axes[0], palette='pastel')
axes[0].set_title("Before Resampling")
sns.countplot(x=y_train_smote, ax=axes[1], palette='pastel')
axes[1].set_title("After SMOTE")
sns.countplot(x=y_train_under, ax=axes[2], palette='pastel')
axes[2].set_title("After Undersampling")
for ax in axes:
    ax.set_xlabel("MedalBinary")
    ax.set_ylabel("Count")
    ax.grid(axis='y')
plt.suptitle("Class Distribution Before and After Resampling", fontsize=14)
plt.tight_layout()
plt.show(block=False)



# load tuned hyperparameters
with open("best_model_params.json") as f:
    best_params = json.load(f)

results = []
f1_scores = []
fitted_models = {}
colors = sns.color_palette("tab10", 10)
color_idx = 0
X_test_scaled_map = {}

plt.figure(figsize=(8, 6))

resamplers = {
    'SMOTE': (X_train_smote, y_train_smote),
    'Undersampling': (X_train_under, y_train_under)
}

for method_name, (X_res, y_res) in resamplers.items():
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_map[method_name] = X_test_scaled

    logreg = LogisticRegression(**best_params[f'LogisticRegression_{method_name}'], class_weight='balanced')
    rf = RandomForestClassifier(**best_params[f'RandomForest_{method_name}'], class_weight='balanced', random_state=15)
    xgb = XGBClassifier(**best_params[f'XGBoost_{method_name}'], use_label_encoder=False, eval_metric='logloss', random_state=15)

    stacking = StackingClassifier(
        estimators=[('lr', logreg), ('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression(),
        cv=3,
        passthrough=False
    )

    models = {
        'LogReg': logreg,
        'RF': rf,
        'XGB': xgb,
        'Stacking': stacking
    }

    for key, model in models.items():
        label = f'{key} ({method_name})'
        model.fit(X_res_scaled, y_res)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        results.append({
            'Model': label,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        })

        fitted_models[label] = model
        f1_scores.append((label, f1_score(y_test, y_pred)))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc_score(y_test, y_proba):.2f})', color=colors[color_idx])
        color_idx += 1



# evaluation results
result_df = pd.DataFrame(results)
print("\n==== Evaluation Metrics Summary ====")
print(result_df.round(3))



# ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower left', fontsize=7)
plt.grid()
plt.tight_layout()
plt.show(block=False)



# confusion matrices SMOTE
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, label in zip(axes.flat, [k for k in fitted_models if '(SMOTE)' in k]):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        fitted_models[label].predict(X_test_scaled_map['SMOTE']),
        ax=ax, cmap='Blues'
    )
    ax.set_title(label)
plt.suptitle("Confusion Matrices – SMOTE Models", fontsize=14)
plt.tight_layout()
plt.show(block=False)



# confusion matrices - undersampling
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, label in zip(axes.flat, [k for k in fitted_models if '(Undersampling)' in k]):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        fitted_models[label].predict(X_test_scaled_map['Undersampling']),
        ax=ax, cmap='Blues'
    )
    ax.set_title(label)
plt.suptitle("Confusion Matrices – Undersampling Models", fontsize=14)
plt.tight_layout()
plt.show(block=False)



# SHAP on XGB (SMOTE)
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_smote)
X_test_scaled = scaler_final.transform(X_test)

xgb_model = XGBClassifier(**best_params['XGBoost_SMOTE'], use_label_encoder=False, eval_metric='logloss', random_state=15)
xgb_model.fit(X_train_scaled, y_train_smote)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
plt.tight_layout()
plt.show(block=False)

input("Stop")
