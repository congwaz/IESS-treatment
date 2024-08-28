import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from flaml import AutoML
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, accuracy_score
import shap
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.utils import resample
import statsmodels.api as sm
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - interval, mean + interval

results = []
roc_data = []
start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(start_time, exist_ok=True)

data_file = "./datasets/treatment2.csv"
data = pd.read_csv(data_file)
data = data.drop(columns=['ID'])
X = data.drop(columns=['Treatment effect'])
y = data['Treatment effect']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X_imputed)
means = standardizer.mean_
stds = standardizer.scale_

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_standardized, y)
smote_data = pd.DataFrame(X_resampled, columns=X.columns)
smote_data['Treatment effect'] = y_resampled
smote_data.to_csv(os.path.join(start_time, 'smote.csv'), index=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0

for train_index, test_index in skf.split(X_resampled, y_resampled):
    fold += 1
    X_train_fold, X_test_fold = X_resampled[train_index], X_resampled[test_index]
    y_train_fold, y_test_fold = y_resampled[train_index], y_resampled[test_index]

    train_fold_df_standardized = pd.DataFrame(X_train_fold, columns=X.columns)
    train_fold_df_standardized['Treatment effect'] = y_train_fold.values
    train_fold_df_standardized.to_csv(os.path.join(start_time, f'standardized_train_fold_{fold}.csv'), index=False)

    test_fold_df_standardized = pd.DataFrame(X_test_fold, columns=X.columns)
    test_fold_df_standardized['Treatment effect'] = y_test_fold.values
    test_fold_df_standardized.to_csv(os.path.join(start_time, f'standardized_test_fold_{fold}.csv'), index=False)

    X_train_fold_original = (X_train_fold * stds) + means
    X_test_fold_original = (X_test_fold * stds) + means

    train_fold_df_original = pd.DataFrame(X_train_fold_original, columns=X.columns)
    train_fold_df_original['Treatment effect'] = y_train_fold.values
    train_fold_df_original.to_csv(os.path.join(start_time, f'train_fold_{fold}.csv'), index=False)

    test_fold_df_original = pd.DataFrame(X_test_fold_original, columns=X.columns)
    test_fold_df_original['Treatment effect'] = y_test_fold.values
    test_fold_df_original.to_csv(os.path.join(start_time, f'test_fold_{fold}.csv'), index=False)

folder_name = start_time

loaded_data = []
results = []

automl = AutoML()
automl_settings = {
    "time_budget": 30,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": os.path.join(start_time, f'flaml_fold{fold}.log'),
    "n_splits": 5,
    "seed": 42,
    "estimator_list": ["xgboost"],  # Only search XGBoost models
}

for i in range(5):
    fold = i + 1
    train_data_file = f"./{folder_name}/standardized_train_fold_{fold}.csv"
    train_data = pd.read_csv(train_data_file)
    X_train = train_data.drop(columns=['Treatment effect']).values
    y_train = train_data['Treatment effect'].values

    val_data_file = f"./{folder_name}/standardized_test_fold_{fold}.csv"
    val_data = pd.read_csv(val_data_file)
    X_val = val_data.drop(columns=['Treatment effect']).values
    y_val = val_data['Treatment effect'].values

    automl.fit(X_train, y_train, **automl_settings)

    y_pred = automl.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    y_pred_proba = automl.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    conf_matrix = confusion_matrix(y_val, y_pred)

    joblib.dump(automl, os.path.join(start_time, f'best_model_fold{fold}.pkl'))

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    bootstrapped_scores = []
    for _ in range(1000):
        indices = resample(np.arange(len(y_pred_proba)), replace=True)
        if len(np.unique(y_val[indices])) < 2:
            continue
        score = roc_auc_score(y_val[indices], y_pred_proba[indices])
        bootstrapped_scores.append(score)

    mean_auc, lower_auc, upper_auc = compute_confidence_interval(bootstrapped_scores)

    results.append({
        'fold': fold,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'auc_95_ci_lower': lower_auc,
        'auc_95_ci_upper': upper_auc,
        'confusion_matrix': conf_matrix
    })

    # Print fold results
    print(f"Fold {fold} results:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"AUC 95% CI: [{lower_auc}, {upper_auc}]")
    print("\n")

    roc_data.extend(zip([fold] * len(fpr), fpr, tpr))

    # SHAP
    best_model = automl.model.estimator

    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_val)
    shap.summary_plot(shap_values, X_val, show=False)
    plt.savefig(os.path.join(start_time, f'shap_summary_fold{fold}.png'))
    plt.close()

    # Nomogram
    logit_model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
    nomogram = logit_model.params

    nomogram_df = pd.DataFrame({
        'Feature': ['Intercept'] + X.columns.tolist(),
        'Coefficient': nomogram
    })

    nomogram_df.to_csv(os.path.join(start_time, f'nomogram_fold{fold}.csv'), index=False)

    # Plot Nomogram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(nomogram_df['Feature'], nomogram_df['Coefficient'], color='skyblue')
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Nomogram for Fold {fold}')
    for i in range(len(nomogram_df['Coefficient'])):
        ax.text(nomogram_df['Coefficient'][i], i, round(nomogram_df['Coefficient'][i], 2), color='black', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(start_time, f'nomogram_fold{fold}.png'))
    plt.close()

roc_df = pd.DataFrame(roc_data, columns=['fold', 'fpr', 'tpr'])
roc_df.to_csv(os.path.join(start_time, 'roc_data.csv'), index=False)
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(start_time, 'results.csv'), index=False)
