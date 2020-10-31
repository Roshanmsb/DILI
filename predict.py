import pickle
from argparse import ArgumentParser
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--saved_model_file", type=str, default='/home/roshan/DILI/csv_files/trained_xgb.pkl',
                        help="path to saved pickle file")
    parser.add_argument("--smiles_csv", type=str, default='/home/roshan/DILI/csv_files/test.csv',
                        help="path to csv file containing smiles stored under SMILES column")
    model_load = parser.parse_args('')
    return model_load


model_args = parser_args()
loaded_model = pickle.load(open(model_args.saved_model_file, "rb"))
df = pd.read_csv(model_args.smiles_csv)
smiles = df['SMILES']
final_predictions = []
print('Starting predictions...')
for i , smile_string in enumerate(smiles):
    try:
        mol = Chem.MolFromSmiles(smile_string)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            2,
            nBits=2048,
            useChirality=False,
            useBondTypes=True,
            useFeatures=False)
        fp = np.asarray(fp, dtype=np.float).reshape((1,-1))
    except:
        print('Invalid smiles string %s' % smile_string)
        continue
    prediction = loaded_model.predict(fp)
    final_predictions.append(prediction)
    if prediction < 0.5:
        print('The smiles string %s at index %d is not toxic to the liver' % (smile_string, i))
    else:
        print('The smiles string %s at index %d is toxic to the liver' % (smile_string, i))
print('Predictions saved under current directory...')
np.save('Predictions', np.array(final_predictions))
