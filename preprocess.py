import deepchem as dc
import numpy as np
from argparse import ArgumentParser


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--train_csv_file", type=str, default='/home/roshan/DILI/csv_files/train.csv',
                        help="path to training data csv file. The Smiles field should be stored in SMILES "
                             "column and labels under DILIConcern column")
    parser.add_argument("--test_csv_file", type=str, default='/home/roshan/DILI/csv_files/test.csv',
                        help="path to test data csv file. The Smiles field should be stored in SMILES "
                             "column and labels under DILIConcern column")
    parser.add_argument("--save_directory", type=str, default='/home/roshan/DILI/csv_files/',
                        help="directory where dataset is stored")
    csv_params = parser.parse_args()
    return csv_params


print("Preprocessing started...")
csv_args = parser_args()
loader = dc.data.CSVLoader(["DILIConcern"], feature_field="SMILES", featurizer=dc.feat.CircularFingerprint())
train_dataset = loader.create_dataset(csv_args.train_csv_file)
test_dataset = loader.create_dataset(csv_args.test_csv_file)

X_train = train_dataset.X
y_train = train_dataset.y
X_test = test_dataset.X
y_test = test_dataset.y

np.save(csv_args.save_directory+'X_train_circular', X_train)
np.save(csv_args.save_directory+'y_train_circular', y_train)
np.save(csv_args.save_directory+'X_test_circular', X_test)
np.save(csv_args.save_directory+'y_test_circular', y_test)
print("Preprocessing done... Datasets saved in %s" % csv_args.save_directory)
