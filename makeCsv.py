"""
This file converts TabDDPm outputs back to csv.
"""

import numpy as np
import pandas as pd
import os
import json

sampled_dirs = ['exp/Shoppers/check', 'exp/indian_liver_patient/check']
column_names_path = ['data/Shoppers/col_names.json','data/indian_liver_patient/col_names.json']
sample_each_dataset = 10

for sampled_dir, column_name_path in zip(sampled_dirs, column_names_path):
    print(sampled_dir, column_name_path)
    for test_idx in range(1, sample_each_dataset+1):
        test_dir = os.path.join(sampled_dir, str(test_idx))
        
        y_train = np.load(os.path.join(test_dir, 'y_train.npy')).reshape(-1, 1)
        if not os.path.exists(os.path.join(test_dir, 'X_num_train.npy')):
            print("No num!")
            X_cat_train = np.load(os.path.join(test_dir, 'X_cat_train.npy'),allow_pickle=True)
            print(X_cat_train.shape,  y_train.shape)
            fake_df = np.concatenate([X_cat_train,y_train], axis=1)
        elif not os.path.exists(os.path.join(test_dir, 'X_cat_train.npy')):
            print("No cat")
            X_num_train = np.load(os.path.join(test_dir, 'X_num_train.npy'))
            print(X_num_train.shape,  y_train.shape)
            fake_df = np.concatenate([X_num_train,y_train], axis=1)
        else:
            X_num_train = np.load(os.path.join(test_dir, 'X_num_train.npy'))
            X_cat_train = np.load(os.path.join(test_dir, 'X_cat_train.npy'),allow_pickle=True)
            print(X_cat_train.shape, X_num_train.shape,  y_train.shape)
            fake_df = np.concatenate([X_cat_train,X_num_train,y_train], axis=1)
        fake_df = pd.DataFrame(fake_df)
        print(fake_df.shape)
        with open(column_name_path, 'r') as column_name_path_file:
            column_names = json.load(column_name_path_file)
        inverse_target_mapping = column_names['inverse_mapping']
        #print(column_names['cat'],column_names['num'],[column_names['target']])
        fake_df.columns = column_names['cat'] + column_names['num'] + [column_names['target']]
        test_name = column_names['test_name'].replace('.csv', '')
        # Reverse cateogry index back to text
        if inverse_target_mapping is not None:
            inverse_target_mapping = {int(k): v for k, v in inverse_target_mapping.items()}
            fake_df[column_names['target']] = fake_df[column_names['target']].map(inverse_target_mapping)
        fake_df.to_csv(os.path.join(sampled_dir, test_name+f'_{test_idx}.csv'),index=False)
