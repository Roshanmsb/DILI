import numpy as np
from sklearn.model_selection import train_test_split
from numpy import mean
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from train import parser_args
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef


print('Loading datasets...')
load_args = parser_args()
X_train = np.load(load_args.load_directory+'X_train_circular.npy')
y_train = np.load(load_args.load_directory+'y_train_circular.npy')
model = XGBClassifier(n_estimators=100, random_state=1, learning_rate=0.1, scale_pos_weight=0.5,
                      max_depth=5, min_child_weight=0.4, gamma=0.8, colsample_bylevel=1, subsample=0.95,
                      colsample_bytree=1, reg_lambda=0.05, reg_alpha=0.7)
cross_validator = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
specificity = make_scorer(recall_score, pos_label=0)
sensitivity = make_scorer(recall_score, pos_label=1)
mathews = make_scorer(matthews_corrcoef)
scoring_metrics = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'F1_score': 'f1_weighted',
                   'Specificity': specificity, 'Sensitivity': sensitivity, 'Mathews_coef': mathews}
print('Starting Cross validation')
scores = cross_validate(model, X_train, y_train, scoring=scoring_metrics,
                        cv=cross_validator, n_jobs=-1)
print('Mean ROC AUC: %.5f' % mean(scores['test_AUC']))
print('Mean Accuracy: %.5f' % mean(scores['test_Accuracy']))
print('Mean F1 score: %.5f' % mean(scores['test_F1_score']))
print('Mean Specificity: %.5f' % mean(scores['test_Specificity']))
print('Mean Sensitivity: %.5f' % mean(scores['test_Sensitivity']))
print('Mean Mathews correlation coefficient: %.5f' % mean(scores['test_Mathews_coef']))
