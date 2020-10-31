import numpy as np
from xgboost import XGBClassifier
from argparse import ArgumentParser
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, matthews_corrcoef


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--load_directory", type=str, default='/home/roshan/DILI/csv_files/',
                        help="directory where dataset is stored")
    parser.add_argument("--save_model_directory", type=str, default='/home/roshan/DILI/csv_files/',
                        help="directory where dataset is stored")
    load_params = parser.parse_args()
    return load_params


if __name__ == '__main__':
    print('Loading datasets...')
    load_args = parser_args()
    X_train = np.load(load_args.load_directory+'X_train_circular.npy')
    y_train = np.load(load_args.load_directory+'y_train_circular.npy')
    X_test = np.load(load_args.load_directory+'X_test_circular.npy')
    y_test = np.load(load_args.load_directory+'y_test_circular.npy')
    print('Datasets loaded... Starting training')
    model = XGBClassifier(n_estimators=100, random_state=1, learning_rate=0.1, scale_pos_weight=0.5,
                          max_depth=5, min_child_weight=0.4, gamma=0.8, colsample_bylevel=1, subsample=0.95,
                          colsample_bytree=1, reg_lambda=0.05, reg_alpha=0.7)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=True, early_stopping_rounds=10)
    y_pred_auc = model.predict_proba(X_test)[:,1]
    print('ROC-AUC %f' % roc_auc_score(y_test, y_pred_auc, average='micro'))
    y_pred = model.predict(X_test)
    print('Accuracy %f' % accuracy_score(y_test, y_pred))
    print('Specificity %f' % recall_score(y_test, y_pred, pos_label=0))
    print('Sensitivity %f' % recall_score(y_test, y_pred, pos_label=1))
    print('F1_score %f' % f1_score(y_test, y_pred,average='weighted'))
    print('Mathews correlation coefficient %f' % matthews_corrcoef(y_test, y_pred))
    print('Training completed... Saving model to %s' % load_args.save_model_directory)
    pickle.dump(model, open(load_args.save_model_directory+"trained_xgb.pkl", "wb"))
