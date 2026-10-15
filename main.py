import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, ConfusionMatrixDisplay
)

from common import load_data, split_train_test, make_sampler, make_pipeline, FEATURES, RANDOM_STATE

os.makedirs('figures', exist_ok=True)

# load dataset
df = load_data()


# split train/test
train_df, test_df = split_train_test(df)

X_train = train_df[FEATURES]
y_train = train_df['MedalBinary']
X_test = test_df[FEATURES]
y_test = test_df['MedalBinary']



# plot class distributions (models resample inside their own pipelines)
_, y_train_smote = make_sampler('SMOTE').fit_resample(X_train, y_train)
_, y_train_under = make_sampler('Undersampling').fit_resample(X_train, y_train)

fig, axes = plt.subplots(3, 1, figsize=(6, 10))
for ax, y, title in zip(axes, [y_train, y_train_smote, y_train_under],
                        ["Before Resampling", "After SMOTE", "After Undersampling"]):
    sns.countplot(x=y, hue=y, ax=ax, palette='pastel', legend=False)
    ax.set_title(title)
    ax.set_xlabel("MedalBinary")
    ax.set_ylabel("Count")
    ax.grid(axis='y')
fig.suptitle("Class Distribution Before and After Resampling", fontsize=14)
fig.tight_layout()
fig.savefig('figures/Figure_1.png')



# load tuned hyperparameters (written by tune.py)
with open("best_model_params.json") as f:
    best_params = json.load(f)

results = []
fitted_models = {}
colors = sns.color_palette("tab10", 10)
color_idx = 0

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for method_name in ['SMOTE', 'Undersampling']:
    def build(model):
        return make_pipeline(model, method_name, best_params[f'{model}_{method_name}'])

    # each base model resamples inside its own pipeline, so the stacking
    # meta-learner is trained on out-of-fold predictions for real rows only
    stacking = StackingClassifier(
        estimators=[('lr', build('LogisticRegression')), ('rf', build('RandomForest')), ('xgb', build('XGBoost'))],
        final_estimator=LogisticRegression(),
        cv=3,
        passthrough=False,
        n_jobs=-1
    )

    models = {
        'LogReg': build('LogisticRegression'),
        'RF': build('RandomForest'),
        'XGB': build('XGBoost'),
        'Stacking': stacking
    }

    for key, pipeline in models.items():
        label = f'{key} ({method_name})'
        # pick the F1-maximising threshold on a held-out 20% of the training
        # data (not resampled), then refit on all training data
        model = TunedThresholdClassifierCV(pipeline, scoring='f1', cv=0.2, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            'Model': label,
            'Threshold': model.best_threshold_,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba),
            'PR AUC': average_precision_score(y_test, y_proba),
            'Accuracy': accuracy_score(y_test, y_pred)
        })
        print(f"Fitted {label}", flush=True)

        fitted_models[label] = model

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f'{label} (AUC={roc_auc_score(y_test, y_proba):.2f})', color=colors[color_idx])
        color_idx += 1



# evaluation results, with an "always predict no medal" baseline for reference
baseline_pred = [0] * len(y_test)
baseline = {
    'Model': 'Baseline (always no medal)',
    'Threshold': float('nan'),
    'Precision': 0.0,
    'Recall': 0.0,
    'F1 Score': 0.0,
    'ROC AUC': 0.5,
    'PR AUC': y_test.mean(),
    'Accuracy': accuracy_score(y_test, baseline_pred)
}
result_df = pd.DataFrame([baseline] + results)
print("\n==== Evaluation Metrics Summary ====")
print(result_df.round(3).to_string(index=False))



# ROC curve
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_title("ROC Curves for All Models")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc='lower left', fontsize=7)
ax_roc.grid()
fig_roc.tight_layout()
fig_roc.savefig('figures/Figure_2.png')



# confusion matrices
for fig_num, method_name in [(3, 'SMOTE'), (4, 'Undersampling')]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, label in zip(axes.flat, [k for k in fitted_models if f'({method_name})' in k]):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            fitted_models[label].predict(X_test),
            ax=ax, cmap='Blues'
        )
        ax.set_title(label)
    fig.suptitle(f"Confusion Matrices – {method_name} Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(f'figures/Figure_{fig_num}.png')



# F1 scores
model_results = result_df.iloc[1:]
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(data=model_results, x='F1 Score', y='Model', hue='Model', palette='Blues_d', legend=False, ax=ax)
ax.set_title("F1 Scores by Model")
ax.set_xlim(0, 1)
ax.set_ylabel("")
ax.grid(axis='x')
fig.tight_layout()
fig.savefig('figures/Figure_5.png')



# SHAP on XGB (SMOTE), reusing the fitted model
xgb_model = fitted_models['XGB (SMOTE)'].estimator_.named_steps['clf']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig('figures/Figure_6.png')

plt.show()
