import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score

# STAGE 1 #
## Predict how much a patient will cost (regression) and who will cost "alot" (classification) ##


# Load data
ins_costs = pd.read_csv(r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\insurance_costs\insurance.csv")

# Data inspection
print(ins_costs.info())
print("Total missing values: ", ins_costs.isna().sum())

## Transformation
### Convert sex, smoker to int (0s, 1s)
ins_costs['sex'] = ins_costs['sex'].map({'female': 0, 'male': 1})
ins_costs['smoker'] = ins_costs['smoker'].map({'yes': 1, 'no': 0})
### Numerise region (OHE)
ins_costs['region'] = ins_costs['region'].map({'southwest': 225, 'northwest': 315, 'southeast': 135, 'northeast': 45})

# Predicting Charges
## Feature Eng. + train/test split
X = ins_costs.drop('charges', axis=1)
y = ins_costs['charges']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=2)

## Selecting model - Ridge Regression
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
solvers = ['auto', 'svd', 'cholesky', 'lsqr']

best_score = 0
best_params = {}

for alpha in alphas:
    for solver in solvers:
        try:
            model = Ridge(alpha=alpha, solver=solver)
            scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')
            mean_score = scores.mean()
            print(f"Alpha: {alpha}, Solver: {solver}, R2 Score: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'alpha': alpha, 'solver': solver}
        except Exception as e:
            print(f"Skipping alpha={alpha}, solver={solver} due to error: {e}")

print('Best params: ', best_params, 'Best R2 score: ', best_score)
model = Ridge(**best_params)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
print("Test R2 score: ", r2_score(y_te, preds))

# Predicting who will cost 'alot'

### Create binary target
### Convert threshold to array for filtering indices
threshold = y.quantile(0.75)  # Top 25% are "a lot"
y_binary = (y >= threshold).astype(int)

### Splits aligned with X and y_binary
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X.reset_index(drop=True),
    y_binary.reset_index(drop=True),
    test_size=0.25,
    random_state=2
)

## Optimized RandomForest parameter search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

best_score_rf = 0
best_params_rf = {}

for n in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        for ss in param_grid['min_samples_split']:
            clf = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_split=ss,
                random_state=2
            )
            scores = cross_val_score(clf, X, y_binary, cv=3, scoring='accuracy')
            mean_score = scores.mean()
            if mean_score > best_score_rf:
                best_score_rf = mean_score
                best_params_rf = {'n_estimators':n, 'max_depth':depth, 'min_samples_split':ss}

print(f"Best params: {best_params_rf}, Score: {best_score_rf:.4f}")

### FInally predicting who will cost alot
final_clf = RandomForestClassifier(max_depth=10, min_samples_split=2, random_state=2)
final_clf.fit(X_tr, y_train_binary)
preds_clf = final_clf.predict(X_test)
#### Convert preds_clf to df for concatenating
preds_df = pd.DataFrame(preds_clf, index=X_te.index, columns=['p75_spender'])
#### Direct assignment
ins_costs['p75_spender'] = np.nan
ins_costs.loc[preds_df.index, 'p75_spender'] = preds_df['p75_spender']
### DropNaN values
insurance_costs = ins_costs.dropna(subset=['p75_spender'])
insurance_costs['p75_spender'] = insurance_costs['p75_spender'].astype(int) # Just ensuring they're 1s and 0s in int


# STAGE 2 #
## Impact Questions (*refer to the impactqs.txt file*)

# STAGE 3 #
## Use patient info to add derived features/ratios ##

### a) Interaction Features that capture non-additive effects
insurance_costs['age_smoker'] = insurance_costs['age'] *( insurance_costs['smoker'] == 1)
#Smoking already a high-risk behavior, but its health impact increases with age
insurance_costs['bmi_smoker'] = insurance_costs['bmi'] *( insurance_costs['smoker']==1)
#  Obesity and smoking both independently increase risk for serious diseases (e.g., heart disease, diabetes).
insurance_costs['age_children'] = insurance_costs['age'] * insurance_costs['children']
# More children mean higher total healthcare costsburden

### b) Risk flags (Binary indicators)
insurance_costs['is_obese'] = insurance_costs['bmi'] >= 20 # Obese people have more complications
insurance_costs['is_senior'] = insurance_costs['age'] >= 60 # Older people have less healthy bodies
insurance_costs['is_expensive_region'] = insurance_costs['region'].isin([45, 135]).astype(int)
# Southeast has poorer health outcomes and higher rates of chronic illnesses, limited Medicaid expansion.
# Northeast has much higher density, demand, cost of living & Medicaid
### c) Ratio features (normalize a feature by another to reveal relative risk)  
insurance_costs['bmi_per_child'] = insurance_costs['bmi'] /( insurance_costs['children'] +1) # Indicates family-wide health risks
insurance_costs['age_per_child'] = insurance_costs['age'] / (insurance_costs['children'] +1)#Older with 1 kid  to younger ones with many kids (different spending)


# STAGE 4 #
## Use train/test split & evaluation metrics that match the aim to reduce false negatives ##

### Evaluate how accurately we predicted patient costs
#### MAE
costs_mae = mean_absolute_error(y_te, preds)
#### MSE
costs_mse = mean_squared_error(y_te, preds)

### Evaluate how accurately the 75th percentile spenders were classified
####F1-score, precision, recall
f1_p75 = f1_score(y_test_binary, preds_clf)
prec_p75 = precision_score(y_test_binary, preds_clf)
recall_p75 = recall_score(y_test_binary, preds_clf)

print(f1_p75) # 0.8511
print(prec_p75) # 0.9475
print(recall_p75) # 0.7792

### Understanding: Model is very confident in who it predicts as "high-cost" but about 22% of actual spenders are missed out on.
current_recall = recall_score(y_test_binary, preds_clf)
## BEcause we want to make sure all patients are treated earlier to prevent long-term complications and cost-volatility, we will raise recall_score
target_recall = 0.9

#f current_recall >= target_recall:
  #print(f"Target achieved: ", current_recall," recall")
#lse:
  #print(f"Target not met: ", current_recall, " recall. Needed ", target_recall)

### Adjust our model's decision threshold to increase recall_score
#### Get exact probabilities
preds_clf = final_clf.predict_proba(X_test)[:, 1]

#### Find threshold that achieves target recall
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test_binary, preds_clf)

if np.any(recall >= target_recall):
    opt_idx = np.argmax(recall >= target_recall)
    opt_thres = thresholds[opt_idx]

    ### Apply new threshold to get better recall
    new_preds_clf = (preds_clf >= opt_thres).astype(int)
    new_recall = recall_score(y_test_binary, new_preds_clf)
    new_prec = precision_score(y_test_binary, new_preds_clf)

    print("New threshold: ", opt_thres)
    print("New recall: ", new_recall)
    print("New precision: ", new_prec)
else:
    print(f"Cannot achieve ", target_recall, " recall")

import matplotlib.pyplot as plt

plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, precision[:-1], label='Precision')
plt.axvline(opt_thres, color='red', linestyle='--')
plt.xlabel("Decision Threshold")
plt.legend()
plt.show()
