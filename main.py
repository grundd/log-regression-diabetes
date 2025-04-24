import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve

# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

pd.set_option('display.max_columns', None)
fsize = 16
tsize = 18

cols_names_fancy = {
    'Diabetes': 'Diabetes',
    'HighBP': 'High blood pressure',
    'HighChol': 'High cholesterol',
    'CholCheck': 'Cholesterol check',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'HeartDiseaseorAttack': 'Hearth disease/attack',
    'PhysActivity': 'Physical activity',
    'Fruits': 'Fruits',
    'Veggies': 'Veggies',
    'HvyAlcoholConsump': 'Heavy alc. consumption',
    'AnyHealthcare': 'Any healthcare',
    'NoDocbcCost': 'No doc due to costs',
    'DiffWalk': 'Difficulties walking',
    'Sex': 'Sex',
    'GenHlth': 'General health',
    'Education': 'Education',
    'Income': 'Income',
    'Age': 'Age',
    'MentHlth': 'Mental health',
    'PhysHlth': 'Physical health',
    'BMI': 'BMI'
}

cols_selected = [
    'GenHlth',
    'HighBP',
    'BMI',
    'HighChol',
    'Age',
    # 'DiffWalk', # removed based on P>|z|
    'Income',
    # 'PhysHlth', # correlated with GenHlth
    'HeartDiseaseorAttack',
    # 'Education', # correlated with Income
    # 'PhysActivity', # removed based on P>|z|
    # 'HvyAlcoholConsump' # removed: "anticorrelated" (probably an unbalanced sample)
    'Veggies',
    'Fruits'
]

feats_binary = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthCare',
    'NoDocbcCost', 'DiffWalk', 'Sex'
]
feats_categories = {
    'GenHlth': [5, 'scale 1-5; 1 = best'],
    'Education': [6, 'scale 1-6; 1 = lowest'],
    'Income': [8, 'scale 1-8; 1 = lowest'],
    'Age': [13, '1 = 18-24; 9 = 60-64; 13 = >80'],
    'MentHlth': [30, '#days of poor mental health in past 30 days'],
    'PhysHlth': [30, '#days of illness/injury in past 30 days'],
}
feats_continuous = [
    'BMI'
]

# function to find correlation of all features with the target
def feat_target_correlation (_x: pd.DataFrame, _y: pd.DataFrame, n_top=0) -> None:
    print('\nCorrelations:')

    # n_top == 0: calculate correlations only for selected columns
    if n_top == 0:
        _x = _x[cols_selected]

    df_corr = pd.concat([_x, _y], axis='columns')
    df_corr.rename(columns=cols_names_fancy, inplace=True)
    correlations = df_corr.corr()['Diabetes'].abs().sort_values(ascending=False)
    corr_to_plot = correlations.drop('Diabetes').index

    # n_top: pick n top correlated cols
    if n_top > 0:
        print(f' Top {n_top} feature correlations with target: \n{correlations.head(n_top)}')
        corr_to_plot = correlations.drop('Diabetes').head(n_top).index

    annot_size = 8
    if len(corr_to_plot) > 15: annot_size = 4

    sns.heatmap(
        df_corr[['Diabetes'] + corr_to_plot.tolist()].corr(),
        annot=True,
        fmt=".2f", # decimal places
        annot_kws={"size": annot_size}, # font size
        cmap='coolwarm'
    )
    plt.title("Top features correlated with diabetes prediction")
    plt.xticks(rotation=45, fontsize=9, ha='right')
    plt.tight_layout()
    if n_top > 0:
        plt.savefig(f'results/correlations_top_{n_top}.pdf')
    else:
        plt.savefig('results/correlations_selected.pdf')
    plt.show()
    return

# function to remove BMI outliers
def remove_bmi_outliers (_x: pd.DataFrame, _y: pd.DataFrame) -> None:
    print('\nRemoving bmi outliers:')

    print(f' #rows before: {len(_x)}')
    _idx = _x[
        (_x['BMI'] < 12.) |
        (_x['BMI'] > 60.)
    ].index
    _x.drop(_idx, inplace=True)
    _y.drop(_idx, inplace=True)
    print(f' #rows after: {len(_x)}')
    return

# function to optimize the threshold
def optimal_thr (_y_test: pd.DataFrame, _y_pred_probs: pd.DataFrame) -> float:
    print('\nFinding optimal threshold:')

    # get precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(_y_test, _y_pred_probs)

    # compute F1 for each threshold
    valid = (precision + recall) > 0
    f1_scores = np.zeros_like(precision)
    f1_scores[valid] = 2 * (precision[valid] * recall[valid]) / (precision[valid] + recall[valid])
    f1_scores = f1_scores[:-1] # because thresholds is shorter by 1, we remove the last F1 score

    # find threshold that gives the best F1
    best_thr = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    print(f' Best threshold: {best_thr:.3f}')
    print(f' Max F1 score: {best_f1:.3f}') # score at best threshold
    print(f' Precision: {precision[np.argmax(f1_scores)]:.3f}')
    print(f' Recall: {recall[np.argmax(f1_scores)]:.3f}')

    # F1 vs. threshold
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='blue')
    plt.axvline(x=best_thr, color='red', linestyle='--', label='Best Threshold')
    plt.title('F1 score vs. classification threshold', fontsize=tsize)
    plt.xlabel('Threshold', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.ylabel('F1 score', fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.grid(True)
    plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.savefig('results/F1_vs_thr.pdf')
    plt.show()

    return best_thr

# function to plot the ROC curve
def roc_and_conf_matrix (
        _y_test: pd.DataFrame, _y_pred_probs: pd.DataFrame, _y_pred_labels: pd.DataFrame) -> None:
    print('\nROC curve and confusion matrix:')

    fpr, tpr, _ = roc_curve(_y_test, _y_pred_probs)
    roc_auc = roc_auc_score(_y_test, _y_pred_probs)

    print(f' Accuracy: {accuracy_score(_y_test, _y_pred_labels):.3f}')
    print(f' ROC AUC: {roc_auc:.3f} (area under curve)')

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Diagonal
    plt.xlabel('False positive rate', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.ylabel('True positive rate', fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.title('ROC curve', fontsize=tsize)
    plt.legend(fontsize=fsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/roc.pdf')
    plt.show()

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    cm_sum = cm.sum()
    cm_percent = cm / cm_sum * 100
    cm_labels = []
    tf_labels = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            tf_lab = tf_labels[i][j]
            row.append(f'{count}\n{percent:.1f}%\n({tf_lab})')
        cm_labels.append(row)
    cm_labels = np.array(cm_labels, dtype=object)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt='', # 'd'
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        annot_kws={"size": 12}, # font size
        cmap="Blues"
    )
    plt.xlabel("Predicted", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.ylabel("Actual", fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.title("Confusion matrix", fontsize=tsize)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.pdf')
    plt.show()

# function to plot the s-curve
def plot_s_curve (feat: str, res, scl, x_train: pd.DataFrame) -> None:
    if feat not in x_train.columns:
        print(f' Not trained for {feat} -> skipping')
        return
    else:
        print(f' Plotting for {feat}')

    feat_min = x_train[feat].min()
    feat_max = x_train[feat].max()

    extra = ''
    if feat in feats_binary:
        # extra = ' (yes/no)'
        original_vals = np.array([0, 1])
    elif feat in feats_categories.keys():
        extra = f' ({feats_categories[feat][1]})'
        original_vals = np.linspace(1, feats_categories[feat][0], feats_categories[feat][0])
    elif feat in feats_continuous:
        original_vals = np.linspace(feat_min, feat_max, 300)
    else:
        print(f' Unknown feature: {feat}')
        return

    # create a DataFrame where all other features are set to their mean
    avg_patient = x_train.mean().copy()
    df_curve = pd.DataFrame([avg_patient] * len(original_vals))
    df_curve[feat] = original_vals

    # scale it using the scaler
    df_curve_scaled = pd.DataFrame(
        scl.transform(df_curve),
        columns=x_train.columns
    )

    # add intercept term (constant) as required by statsmodels
    df_curve_scaled_const = sm.add_constant(df_curve_scaled, has_constant='add')

    # predict using the trained model
    predicted_probs = res.predict(df_curve_scaled_const)

    # plot the sigmoid (S) curve
    plt.figure(figsize=(5, 4))
    if feat in feats_continuous:
        plt.plot(original_vals, predicted_probs, color='blue')
        plt.xticks(fontsize=fsize)
    else:
        plt.plot(original_vals, predicted_probs, color='blue', linestyle='--', marker='o')
        if feat in feats_binary:
            plt.xticks([0, 1], labels=['No', 'Yes'], fontsize=fsize)
            plt.xlim(-.25, 1.25)
        else:
            plt.xticks(original_vals, labels=[f"{x:.0f}" for x in original_vals], fontsize=fsize)
    plt.title(f'{cols_names_fancy[feat]}', fontsize=tsize)
    plt.xlabel(f"{cols_names_fancy[feat]}{extra}", fontsize=fsize)
    plt.ylabel("Predicted prob. of diabetes", fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    # plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(f'results/Scurve_{feat}.pdf')
    plt.show()
    return

def simulate_lifestyle_adjustment (_x_test: pd.DataFrame, _y_test: pd.DataFrame, best_thr: float) -> None:
    # filter a subset of patients
    test = pd.concat([_x_test, _y_test], axis='columns')
    filtered = test[
        (test['BMI'] > 25.) &
        (test['BMI'] < 30.) &
        (test['Veggies'] == 0) &
        (test['Fruits'] == 0) &
        (test['Diabetes'] == 1)
        ]
    x_filtered = filtered.drop(columns=['Diabetes'])
    print(f'\nFiltered test dataset #rows: {len(filtered)}')

    # predict baseline risk
    scaled = pd.DataFrame(scaler.transform(x_filtered), columns=X_test.columns)
    scaled_const = sm.add_constant(scaled, has_constant='add')
    scaled_const = scaled_const[result.model.exog_names]
    baseline_risk = result.predict(scaled_const)

    # simulate healthy behavior change
    adjusted = x_filtered.copy()
    adjusted["BMI"] = (adjusted["BMI"] - 5)  # reduce BMI
    adjusted["Veggies"] = 1
    adjusted["Fruits"] = 1
    adjusted["GenHlth"] = np.maximum(1, adjusted["GenHlth"] - 1)

    # scale and predict again
    scaled_new = pd.DataFrame(scaler.transform(adjusted), columns=X_test.columns)
    scaled_new_const = sm.add_constant(scaled_new, has_constant='add')
    scaled_new_const = scaled_new_const[result.model.exog_names]
    new_risk = result.predict(scaled_new_const)
    diff = new_risk - baseline_risk

    print(f' Mean before: {baseline_risk.mean():.3f}')
    print(f' Mean after: {new_risk.mean():.3f}')

    # try out a single patient
    idx = 0
    print(f' Chosen patient: idx {idx}')
    patient = filtered.iloc[idx]
    print(f' Patient: \n{patient}')
    print(f' Baseline diabetes risk: {baseline_risk[0]:.3f}')
    patient_intervened = adjusted.iloc[idx]
    print(f' Intervened patient: \n{patient_intervened}')
    print(f' New diabetes risk: {new_risk[0]:.3f}')

    width = 0.05
    binning = np.arange(0, 1 + width, width)
    plt.figure(figsize=(6, 5))
    h1, _, _ = plt.hist(baseline_risk, color='red', bins=binning, alpha=0.5, label='Baseline diabetes risk')
    h2, _, _ = plt.hist(new_risk, color='blue', bins=binning, alpha=0.5, label='Diabetes risk after adjustments')
    plt.axvline(x=best_thr, color='black', linestyle='--', label=f'Threshold ({best_thr:.3f})')
    plt.xlabel('Diabetes risk', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.ylabel('Counts', fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylim(0, max(h1.max(), h2.max()) * 1.4)
    plt.title('Lifestyle adjustment on a filtered dataset', fontsize=tsize)
    plt.legend(fontsize=13, framealpha=1)
    plt.tight_layout()
    plt.savefig(f'results/lifestyle_adjust.pdf')
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.hist(diff, color='blue', alpha=0.5)
    plt.title('Risk change after lifestyle adjustment', fontsize=tsize)
    plt.xlabel('New risk \N{MINUS SIGN} baseline risk', fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.ylabel('Counts', fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.savefig(f'results/lifestyle_adjust_difference.pdf')
    plt.show()
    return

df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
df.rename(columns={'Diabetes_binary': 'Diabetes'}, inplace=True)

print(f'\n{df.describe()}')
print(f'\n{df["Diabetes"].value_counts()}')

# histogram & box plot for BMI
plt.hist(df['BMI'], bins=50, color='blue', alpha=0.5)
plt.title('BMI histogram', fontsize=tsize)
plt.xlabel('BMI', fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.ylabel('Counts', fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.yscale('log')
plt.tight_layout()
plt.savefig('results/BMI_histogram.pdf')
plt.show()

plt.boxplot(df['BMI'], whis=1.5)
plt.title('BMI Box Plot')
plt.savefig('results/BMI_boxplot.pdf')
plt.show()

# define X (features) and y (target)
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']

# find the best features correlated with the target
feat_target_correlation(X, y)
feat_target_correlation(X, y, 10)
feat_target_correlation(X, y, 21)

# define X and y taking the most promising features
X = X[cols_selected]

# split the dataset into 'train' (80%) and 'test' (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # stratify=y,

# filter out possible BMI outliers in X_train
remove_bmi_outliers(X_train, y_train)

# scale the train and test datasets
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# add a constant (intercept), required when using statsmodels
X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)

# fit the model
logit_model = sm.Logit(y_train, X_train_scaled_const)
result = logit_model.fit()

# display full summary
print(f'\nSummary:\n{result.summary()}')

pvals = result.pvalues

# define conditions
conditions = [
    pvals > 0.05,
    (pvals <= 0.05) & (pvals > 0.01),
    (pvals <= 0.01) & (pvals > 0.001),
    pvals <= 0.001
]

# define the corresponding categories
choices = ['-', '*', '**', '***']

significance = np.select(conditions, choices, default='-')

summary_table = pd.DataFrame({
    'Parameter': result.params.index.tolist(),
    'Coefficient': [ f'{coef:.2f} \u00B1 {sdev:.2f}' for coef, sdev in zip(result.params, result.bse) ],
    'z-score': [ f'{zscore:.2f}' for zscore in result.tvalues ],
    'p-value': significance
})

print(summary_table.to_csv(sep='\t', index=False))

explicit_pvals = pd.DataFrame({
    'Coefficient': result.params.index.tolist(),
    'p-value': result.pvalues
})

print(f"\n{explicit_pvals}")

# predict probabilities
y_pred_probs = result.predict(X_test_scaled_const)

# find optimal threshold and classify the results
best_threshold = optimal_thr(y_test, y_pred_probs)
y_pred_labels = (y_pred_probs >= best_threshold).astype(int)

# print the ROC curve and the confusion matrix
roc_and_conf_matrix(y_test, y_pred_probs, y_pred_labels)

# plot S curves
print('\nS-curves:')
for f in feats_continuous + feats_binary:
    plot_s_curve(f, result, scaler, X_train)
for f in feats_categories.keys():
    plot_s_curve(f, result, scaler, X_train)

# lifestyle adjustments
simulate_lifestyle_adjustment(X_test, y_test, best_threshold)
