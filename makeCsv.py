"""
This file converts TabDDPm outputs back to csv.
"""

import numpy as np
import pandas as pd
import os
import json

#sampled_dirs = ['exp/abalone/check', 'exp/adult/check', 'exp/churn2/check', 'exp/insurance/check', 'exp/wilt/check']
sampled_dirs = ['exp/abalone/check']
column_names_path = ['data/abalone/col_names.json']
sample_each_dataset = 2

for sampled_dir, column_name_path in zip(sampled_dirs, column_names_path):
    for test_idx in range(1, sample_each_dataset+1):
        test_dir = os.path.join(sampled_dir, str(test_idx))
        X_cat_train = np.load(os.path.join(test_dir, 'X_cat_train.npy'),allow_pickle=True)
        X_num_train = np.load(os.path.join(test_dir, 'X_num_train.npy'))
        y_train = np.load(os.path.join(test_dir, 'y_train.npy')).reshape(-1, 1)
        print(X_cat_train.shape, X_num_train.shape,  y_train.shape)
        fake_df = np.concatenate([X_cat_train,X_num_train,y_train], axis=1)
        fake_df = pd.DataFrame(fake_df)
        with open(column_name_path, 'r') as column_name_path_file:
            column_names = json.load(column_name_path_file)
        inverse_target_mapping = column_names['inverse_mapping']
        fake_df.columns = column_names['cat'] + column_names['num'] + [column_names['target']]
        test_name = column_names['test_name'].replace('.csv', '')
        # Reverse cateogry index back to text
        if inverse_target_mapping is not None:
            inverse_target_mapping = {int(k): v for k, v in inverse_target_mapping.items()}
            fake_df[column_names['target']] = fake_df[column_names['target']].map(inverse_target_mapping)
        fake_df.to_csv(os.path.join(sampled_dir, test_name+f'_{test_idx}.csv'),index=False)
