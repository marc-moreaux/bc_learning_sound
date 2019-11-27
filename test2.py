from argparse import Namespace
from shutil import copy
from guppy import hpy
import pickle as pkl
import pandas as pd
import numpy as np
import glob
import test
import os


def get_rand_args():
    args = Namespace(
        save='./results/esc50_gap_7',
        split=[1, 2, 3],
        noiseAugment=bool(np.random.randint(2)),
        inputLength=0,
        act_thrld=np.random.uniform(1, 50),
        act_window=np.random.randint(5, 10),
        min_act_per_window=np.random.randint(2, 5),
        use_zero_activations=bool(np.random.randint(2)),
        n_batch_eval=100,
        store_cm=True)
    if args.min_act_per_window > args.act_window:
        args.min_act_per_window = args.act_window

    args._str = 'th{}_win{}_actPwin{}_inLen{}_nBatch{}_nAug{}_zeroAct{}'.format(
            args.act_thrld, args.act_window, args.min_act_per_window,
            args.inputLength, args.n_batch_eval, args.noiseAugment, args.use_zero_activations)
    return args


results_name = 'ommission, insertion, acc, FN, TP, FP, TN, conf_matrix, precision, rappel'.split(',')
df_path = os.path.join('./results/exp.csv')
dict_path = os.path.join('./results/exp.pkl')

# create file if not present
if not os.path.isfile(dict_path):
    with open(dict_path, "wb") as f:
        pkl.dump([], f)

with open(dict_path, 'rb') as f:
    mdicts = pkl.load(f)

# Read csv if any
df = None
if os.path.isfile(df_path):
    df = pd.read_csv(df_path)
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

for _ in range(10):
    args = get_rand_args()
    results = test.main2(get_rand_args())
    params = vars(args)
    params['split'] = str(params['split'])
    params.update(results)

    # Update csv
    if df is None:
        params['split'] = [-1]
        df = pd.DataFrame.from_dict(params)
    else:
        df = df.append(params, ignore_index=True)

    params['idx'] = df.index[-1]
    mdicts.append(params)
    with open(dict_path, 'wb') as f:
        pkl.dump(mdicts, f)
    
    # Save csv
    print('-- saving --')
    if os.path.isfile(df_path):
        copy(df_path, df_path + '_old')
    with open(df_path, 'w') as f:
        df.to_csv(f)
        

